import os

def find_checkpoint(checkpoint_dir, checkpoint_num=None):
  found_checkpoint = False
  if os.path.exists(checkpoint_dir):
    print('Found checkpoint directory!')
    checkpoints = sorted(os.listdir(checkpoint_dir))
    if checkpoint_num is not None:
      for checkpoint in checkpoints:
        if str(checkpoint_num) in checkpoint:
          model_name = checkpoint
          found_checkpoint = True
          break
    else:
      model_name = checkpoints[-1]
      found_checkpoint = True
    if found_checkpoint:
      print('Found checkpoint: Loading checkpoint ' + model_name + '...')
    else:
      print('Checkpoint number does not exist! Please choose from:')
      print(checkpoints)
      exit()
  else:
    print('Did not find checkpoint directory! Training from scratch...')
    model_name = None

  return model_name
