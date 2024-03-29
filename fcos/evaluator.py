import os
import colorsys
import cv2
import numpy as np
import torch
import torch.nn as nn
import logging
from torchvision import transforms as T
import matplotlib.pyplot as plt

from fcos.core.data.datasets.evaluation import evaluate
from fcos.core.utils.timer import Timer, get_time_str
from fcos.core.engine.inference import compute_on_dataset
from fcos.core.modeling.roi_heads.mask_head.inference import Masker
from fcos.core.structures.keypoint import PersonKeypoints
from fcos.core.structures.image_list import to_image_list
from fcos.core import layers as L
from fcos.core.utils import cv2_util


class Evaluator(nn.Module):

    # COCO categories for pretty print
    CATEGORIES = [
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    def __init__(self,
                 cfg,
                 show_mask_heatmaps=False,
                 masks_per_dim=2,
                 min_image_size=224):
        super().__init__()

        # The following per-class thresholds are computed by maximizing
        # per-class f-measure in their precision-recall curve.
        # Please see compute_thresholds_for_classes() in coco_eval.py for details.
        confidence_thresholds_for_classes = [
            0.4923645853996277, 0.4928510785102844, 0.5040897727012634,
            0.4912887513637543, 0.5016880631446838, 0.5278812646865845,
            0.5351834893226624, 0.5003424882888794, 0.4955945909023285,
            0.43564629554748535, 0.6089804172515869, 0.666087806224823,
            0.5932040214538574, 0.48406165838241577, 0.4062422513961792,
            0.5571075081825256, 0.5671307444572449, 0.5268378257751465,
            0.5112953186035156, 0.4647842049598694, 0.5324517488479614,
            0.5795850157737732, 0.5152440071105957, 0.5280804634094238,
            0.4791383445262909, 0.5261335372924805, 0.4906163215637207,
            0.523737907409668, 0.47027698159217834, 0.5103300213813782,
            0.4645252823829651, 0.5384289026260376, 0.47796186804771423,
            0.4403403103351593, 0.5101461410522461, 0.5535093545913696,
            0.48472103476524353, 0.5006796717643738, 0.5485560894012451,
            0.4863888621330261, 0.5061569809913635, 0.5235867500305176,
            0.4745445251464844, 0.4652363359928131, 0.4162440598011017,
            0.5252017974853516, 0.42710989713668823, 0.4550687372684479,
            0.4943239390850067, 0.4810051918029785, 0.47629663348197937,
            0.46629616618156433, 0.4662836790084839, 0.4854755401611328,
            0.4156557023525238, 0.4763634502887726, 0.4724511504173279,
            0.4915047585964203, 0.5006274580955505, 0.5124194622039795,
            0.47004589438438416, 0.5374764204025269, 0.5876904129981995,
            0.49395060539245605, 0.5102297067642212, 0.46571290493011475,
            0.5164387822151184, 0.540651798248291, 0.5323763489723206,
            0.5048757195472717, 0.5302401781082153, 0.48333442211151123,
            0.5109739303588867, 0.4077408015727997, 0.5764586925506592,
            0.5109297037124634, 0.4685552418231964, 0.5148998498916626,
            0.4224434792995453, 0.4998510777950287
        ]

        self.cfg = cfg.clone()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.min_image_size = min_image_size
        self.transforms = self.build_transform()

        # segmentation mask
        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.cat_colors = np.apply_along_axis(
            lambda r: colorsys.hsv_to_rgb(r[0], r[1], r[2]),
            axis=1,
            arr=np.c_[
                np.linspace(0, 1, len(Evaluator.CATEGORIES), endpoint=False),
                np.full(len(Evaluator.CATEGORIES), 1),
                np.full(len(Evaluator.CATEGORIES), 0.75)])
        self.cpu_device = torch.device("cpu")
        self.confidence_thresholds_for_classes = torch.tensor(
            confidence_thresholds_for_classes)
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN,
                                          std=cfg.INPUT.PIXEL_STD)

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self.min_image_size),
            T.ToTensor(),
            to_bgr_transform,
            normalize_transform,
        ])
        return transform

    def run_on_opencv_image(self, image, model):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image, model)
        top_predictions = self.select_top_predictions(predictions)

        result = image.copy()
        if self.show_mask_heatmaps and top_predictions.has_field('mask'):
            return self.create_mask_montage(result, top_predictions)
        result = self.overlay_boxes(result, top_predictions)
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions)
        if self.cfg.MODEL.KEYPOINT_ON:
            result = self.overlay_keypoints(result, top_predictions)
        result = self.overlay_class_names(result, top_predictions)

        return result, top_predictions

    def compute_prediction(self, original_image, model):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image,
                                   self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        labels = predictions.get_field("labels")
        thresholds = self.confidence_thresholds_for_classes[(labels -
                                                             1).long()]
        keep = torch.nonzero(scores > thresholds).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        return np.atleast_2d(
            (self.cat_colors[labels[:]] * 255).astype("uint8"))

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()
        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right),
                                  (255, 255, 255), 3)
            image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right),
                                  tuple(color), 2)

        return image

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        colors = self.compute_colors_for_labels(labels).tolist()

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None]
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            image = cv2.drawContours(image, contours, -1, color, 3)

        composite = image

        return composite

    def overlay_keypoints(self, image, predictions):
        keypoints = predictions.get_field("keypoints")
        kps = keypoints.keypoints
        scores = keypoints.get_field("logits")
        kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()
        for region in kps:
            image = vis_keypoints(image, region.transpose((1, 0)))
        return image

    def create_mask_montage(self, image, predictions):
        """
        Create a montage showing the probability heatmaps for each one one of the
        detected objects

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask`.
        """
        masks = predictions.get_field("mask")
        masks_per_dim = self.masks_per_dim
        masks = L.interpolate(masks.float(),
                              scale_factor=1 / masks_per_dim).byte()
        height, width = masks.shape[-2:]
        max_masks = masks_per_dim**2
        masks = masks[:max_masks]
        # handle case where we have less detections than max_masks
        if len(masks) < max_masks:
            masks_padded = torch.zeros(max_masks,
                                       1,
                                       height,
                                       width,
                                       dtype=torch.uint8)
            masks_padded[:len(masks)] = masks
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        result = torch.zeros((masks_per_dim * height, masks_per_dim * width),
                             dtype=torch.uint8)
        for y in range(masks_per_dim):
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[y, x]
        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(
            predictions.get_field("labels")).tolist()

        template = "{}: {:.2f}"
        for box, score, label, color in zip(boxes, scores, labels, colors):
            x, y = box[:2]
            x = int(round(x.item()))
            y = int(round(y.item()))
            s = template.format(label, score)
            cv2.putText(image, s, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, .75,
                        (255, 255, 255), 2)
            cv2.putText(image, s, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, .75,
                        tuple(color), 1)

        return image

    # inference of trained model
    def inference(self,
                  model,
                  data_loader,
                  dataset_name,
                  iou_types=("bbox",),
                  box_only=False,
                  device="cuda",
                  expected_results=(),
                  expected_results_sigma_tol=4,
                  output_folder=None):
        # Ensure output folder exists
        if output_folder is not None:
            os.makedirs(output_folder, exist_ok=True)

        # convert to a torch.device for efficiency
        device = torch.device(device)
        logger = logging.getLogger("fcos_core.inference")
        dataset = data_loader.dataset
        logger.info("Start evaluation on {} dataset({} images).".format(
            dataset_name, len(dataset)))
        total_timer = Timer()
        inference_timer = Timer()
        total_timer.tic()
        predictions = compute_on_dataset(model, data_loader, device,
                                         inference_timer)
        total_time = total_timer.toc()
        total_time_str = get_time_str(total_time)
        logger.info("Total run time: {}".format(total_time_str))
        total_infer_time = get_time_str(inference_timer.total_time)
        logger.info("Model inference time: {})".format(total_infer_time))

        if output_folder:
            torch.save(predictions,
                       os.path.join(output_folder, "predictions.pth"))

        # convert a dict where the key is the index in a list
        image_ids = list(sorted(predictions.keys()))
        if len(image_ids) != image_ids[-1] + 1:
            logger = logging.getLogger("fcos_core.inference")
            logger.warning(
                "Number of images that were gathered from multiple processes is not "
                "a contiguous set. Some images might be missing from the evaluation"
            )

        # convert to a list
        predictions = [predictions[i] for i in image_ids]

        extra_args = dict(
            box_only=box_only,
            iou_types=iou_types,
            expected_results=expected_results,
            expected_results_sigma_tol=expected_results_sigma_tol,
        )

        return evaluate(dataset=dataset,
                        predictions=predictions,
                        output_folder=output_folder,
                        **extra_args)


def vis_keypoints(img, kps, kp_thresh=2, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    dataset_keypoints = PersonKeypoints.NAMES
    kp_lines = PersonKeypoints.CONNECTIONS

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (kps[:2, dataset_keypoints.index('right_shoulder')] +
                    kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
    sc_mid_shoulder = np.minimum(
        kps[2, dataset_keypoints.index('right_shoulder')],
        kps[2, dataset_keypoints.index('left_shoulder')])
    mid_hip = (kps[:2, dataset_keypoints.index('right_hip')] +
               kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
    sc_mid_hip = np.minimum(kps[2, dataset_keypoints.index('right_hip')],
                            kps[2, dataset_keypoints.index('left_hip')])
    nose_idx = dataset_keypoints.index('nose')
    if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
        cv2.line(kp_mask,
                 tuple(mid_shoulder),
                 tuple(kps[:2, nose_idx]),
                 color=colors[len(kp_lines)],
                 thickness=2,
                 lineType=cv2.LINE_AA)
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        cv2.line(kp_mask,
                 tuple(mid_shoulder),
                 tuple(mid_hip),
                 color=colors[len(kp_lines) + 1],
                 thickness=2,
                 lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(kp_mask,
                     p1,
                     p2,
                     color=colors[l],
                     thickness=2,
                     lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(kp_mask,
                       p1,
                       radius=3,
                       color=colors[l],
                       thickness=-1,
                       lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(kp_mask,
                       p2,
                       radius=3,
                       color=colors[l],
                       thickness=-1,
                       lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
