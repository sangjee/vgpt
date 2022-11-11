
from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
import cv2
import albumentations as A

class CustomDataset(Dataset):
  def __init__(self, image_filenames, captions, image_input1, image_input2, image_input3, tokenizer, transforms):
    self.image_filenames = image_filenames
    self.image_input1 = image_input1
    self.image_input2 = image_input2
    self.image_input3 = image_input3
    self.captions = list(captions)
    # self.encoded_captions = tokenizer(
    #     list(captions)
    # )
    self.tokenizer = tokenizer
    self.transforms = transforms
    
  def __len__(self):
    return len(self.captions)

  def __getitem__(self, index: int):
    # item = {
    #   key: torch.tensor(values[index])
    #   for key, values in self.encoded_captions.items()
    # }
    item = {}
    with h5py.File(self.image_filenames[index], 'r') as hf:
        image1 = _get_image(hf.get(self.image_input1[index]))
        image2 = _get_image(hf.get(self.image_input2[index]))
        image3 = _get_image(hf.get(self.image_input3[index]))
        image_3d = np.zeros((256, 256, 3),dtype=np.uint8)
        image_3d[:,:,0] = image1
        image_3d[:,:,1] = image2
        image_3d[:,:,2] = image3
    hf.close()
    image_3d = self.transforms(image=image_3d)['image']
    item['image'] = torch.tensor(image_3d).permute(2, 0, 1).float()
    # item['image'] = item['image'].reshape(-1,224)
    # item['text'] = self.captions[index]
    tmp = self.tokenizer.preprocess(self.captions[index])
    item['text'] = self.tokenizer.process(tmp)

    return item

def _get_image(image):
    image = np.array(image)
    image = image.astype(np.float32)
    # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    # image = cv2.resize(image, dsize=(256, 256))

    return image