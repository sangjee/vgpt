
from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
import cv2
import albumentations as A

class CustomDataset(Dataset):
  def __init__(self, image_filenames, captions, tokenizer, transforms):
    self.image_filenames = image_filenames
    self.captions = list(captions)
    # self.encoded_captions = tokenizer(
    #     list(captions)
    # )
    self.tokenizer = tokenizer
    self.transforms = transforms
    
  def __len__(self):
    return len(self.captions)

  def __getitem__(self, index: int):
    item = {}
    path = {'image':self.image_filenames[index]}
    image_3d = self.transforms(path)
    item['image']=(image_3d['image']).squeeze(0).permute(2, 0, 1).float()
    item['image'] = (item['image']).clone().detach()
    # item['image'] = item['image'].reshape(-1,224)
    # item['text'] = self.captions[index]
    tmp = self.tokenizer.preprocess(self.captions[index])
    item['text'] = self.tokenizer.process(tmp)

    return item