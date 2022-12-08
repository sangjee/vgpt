
from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
import cv2
import albumentations as A
import re

class CustomDataset2(Dataset):
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
    mask_info = dict()
    with h5py.File(self.image_filenames[index], 'r') as hf:
      keys = list(hf.keys())
      keys.sort()
      for i, fName in enumerate(keys):
        if 'label' in fName:
          mask_info[int(re.sub(r'[^0-9]', '', fName))] = np.sum(hf.get(fName))
      sorted_info = sorted(mask_info.items(), key=lambda item: item[1], reverse=True)
      mask_id=[]
      for i in range(20):
        mask_id.append(sorted_info.pop(0))
      image_3d = np.zeros((256, 256, 20),dtype=np.float32) # mask와 이미지곱으로 이미지 생성
      for idx, key in enumerate(mask_id) :
        for i, fName in enumerate(keys):
          if str(mask_id[idx][0]) in fName:
            if 'input' in fName:
              img = _get_image(hf.get(fName),mode='img')
            elif 'label' in fName:
              mask = _get_image(hf.get(fName),mode='mask')
        image_3d[:,:,idx] = img*mask
    hf.close()
    image_3d = self.transforms(image=image_3d)['image']
    item['image'] = torch.tensor(image_3d).permute(2, 0, 1).float()
    # tmp = self.tokenizer.preprocess(self.captions[index])
    item['text'] = self.captions[index]
    return item

def _get_image(image, mode='img'):
    if mode == 'img':
        image = np.array(image)
        image = image.astype(np.float32)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = cv2.resize(image, dsize=(256, 256))
        return image

    elif mode == 'mask':
        image = np.array(image)
        image = cv2.resize(image, dsize=(256, 256))
        return image