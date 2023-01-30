import pytorch_lightning as pl
import albumentations as A
from torch.utils.data import DataLoader

from CustomDataset import CustomDataset

from monai.transforms import (
    LoadImaged,
    Compose,
    RandRotated,
    EnsureChannelFirstd,
    Resized,
    ScaleIntensityd,
    EnsureTyped,
    Spacingd,
    Orientationd
)

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, batch_size, num_workers, tokenizer, mode='train', d_type='mri', channel=20):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.mode = mode
        self.d_type = d_type
        self.channel = channel

    def setup(self, stage=None):
        self.train_dataset = build_loaders(self.train_df, self.tokenizer, self.mode, self.d_type, self.channel)
        self.val_dataset = build_loaders(self.val_df, self.tokenizer, self.mode, self.d_type, self.channel)
        self.test_dataset = build_loaders(self.test_df, self.tokenizer, self.mode, self.d_type, self.channel)
        
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
def build_loaders(dataframe, tokenizer, mode, d_type, channel):
    if d_type=='ct':
        transforms = get_transforms2(mode=mode, channel=channel)
        dataset = CustomDataset(
            dataframe['image'].values,
            dataframe['caption'].values,
            None,
            None,
            None,
            tokenizer=tokenizer,
            transforms=transforms,
            mode=mode,
            d_type=d_type
            )
    else :
        transforms = get_transforms(mode=mode)
        dataset = CustomDataset(
            dataframe['image'].values,
            dataframe['caption'].values,
            dataframe['input_img1'].values,
            dataframe['input_img2'].values,
            dataframe['input_img3'].values,
            tokenizer=tokenizer,
            transforms=transforms,
            mode=mode,
            d_type=d_type
            )
    return dataset
  
def get_transforms(mode="train"):
    if mode == "train":
        # [TODO] add rotate etc...
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

def get_transforms2(mode="train", channel=20):
    if mode == "train":
        return Compose([
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            # RandRotated(keys="image", range_x=np.pi / 12, prob=0.3), 
            ScaleIntensityd(keys="image"),
            Spacingd(keys='image',pixdim=(1,1,5)),
            Resized(keys="image", spatial_size=[128,128,channel]),
            EnsureTyped(keys="image"),
            # Orientationd(keys="image", axcodes="SPL")
        ])
    else:
        return Compose([
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            ScaleIntensityd(keys="image"),
            Spacingd(keys='image',pixdim=(1,1,5)),
            Resized(keys="image", spatial_size=[128,128,channel]),
            EnsureTyped(keys="image"),
            # Orientationd(keys="image", axcodes="SPL")
        ])