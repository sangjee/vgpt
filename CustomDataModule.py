import pytorch_lightning as pl
import albumentations as A
from torch.utils.data import DataLoader

from CustomDataset import CustomDataset
from CustomDataset2 import CustomDataset2

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, batch_size, num_workers, tokenizer, mode='train'):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.mode = mode

    def setup(self, stage=None):
        self.train_dataset = build_loaders(self.train_df, self.tokenizer, self.mode)
        self.val_dataset = build_loaders(self.val_df, self.tokenizer, self.mode)
        self.test_dataset = build_loaders(self.test_df, self.tokenizer, self.mode)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
            )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
            )
def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)

    if mode == "train":
        dataset = CustomDataset(
            dataframe['image'].values,
            dataframe['caption'].values,
            dataframe['input_img1'].values,
            dataframe['input_img2'].values,
            dataframe['input_img3'].values,
            tokenizer=tokenizer,
            transforms=transforms,
            )
        return dataset
    elif mode == "valid":
        dataset = CustomDataset2(
            dataframe['image'].values,
            dataframe['caption'].values,
            dataframe['input_img1'].values,
            dataframe['input_img2'].values,
            dataframe['input_img3'].values,
            tokenizer=tokenizer,
            transforms=transforms,
            )
        return dataset
  
def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(224, 224, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(224, 224, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )