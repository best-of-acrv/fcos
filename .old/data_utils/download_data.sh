#!/bin/bash
# Download COCO dataset
mkdir -p datasets/coco && cd datasets/coco
curl -O http://images.cocodataset.org/zips/train2014.zip
curl -O http://images.cocodataset.org/zips/val2014.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2014.zip
# Unzip contents
echo 'Unzipping contents...'
unzip train2014.zip -d .
unzip val2014.zip -d .
unzip annotations_trainval2014.zip -d .
wget -O annotations/instances_valminusminival2014.json https://cloudstor.aarnet.edu.au/plus/s/dyEimvGtiPiGUCQ/download
wget -O annotations/instances_minival2014.json https://cloudstor.aarnet.edu.au/plus/s/8zMCnu1GQsZFLXy/download
# remove zip files
rm train2014.zip
rm val2014.zip
rm annotations_trainval2014.zip