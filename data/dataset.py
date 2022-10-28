import os
import numpy as np
import itertools
import collections
import torch
from .example import Example
from .utils import nostdout
from pycocotools.coco import COCO as pyCOCO


class Dataset(object):
    def __init__(self, examples, fields):
        self.examples = examples
        self.fields = dict(fields)

    def collate_fnc(self):
        def collate(batch):
            if len(self.fields) == 1:
                batch = [batch, ]
            else:
                batch = list(zip(*batch))

            tensors = []
            for field, data in zip(self.fields.values(), batch):
                tensor = field.process(data)
                if isinstance(tensor, collections.Sequence) and any(isinstance(t, torch.Tensor) for t in tensor):
                    tensors.extend(tensor)
                else:
                    tensors.append(tensor)

            if len(tensors) > 1:
                return tensors
            else:
                return tensors[0]
        return collate

    def __getitem__(self, i):
        example = self.examples[i]
        data = []
        for field_name, field in self.fields.items():
            data.append(field.preprocess(getattr(example, field_name)))

        if len(data) == 1:
            data = data[0]
        return data

    def __len__(self):
        return len(self.examples)

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)


class ValueDataset(Dataset):
    def __init__(self, examples, fields, dictionary):
        self.dictionary = dictionary
        super(ValueDataset, self).__init__(examples, fields)

    def collate_fnc(self):
        def collate(batch):
            value_batch_flattened = list(itertools.chain(*batch))
            value_tensors_flattened = super(ValueDataset, self).collate_fnc()(value_batch_flattened)

            lengths = [0, ] + list(itertools.accumulate([len(x) for x in batch]))
            if isinstance(value_tensors_flattened, collections.Sequence) \
                    and any(isinstance(t, torch.Tensor) for t in value_tensors_flattened):
                value_tensors = [[vt[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])] for vt in value_tensors_flattened]
            else:
                value_tensors = [value_tensors_flattened[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])]

            return value_tensors
        return collate

    def __getitem__(self, i):
        if i not in self.dictionary:
            raise IndexError

        values_data = []
        for idx in self.dictionary[i]:
            value_data = super(ValueDataset, self).__getitem__(idx)
            values_data.append(value_data)
        return values_data

    def __len__(self):
        return len(self.dictionary)


class DictionaryDataset(Dataset):
    def __init__(self, examples, fields, key_fields):
        if not isinstance(key_fields, (tuple, list)):
            key_fields = (key_fields,)
        for field in key_fields:
            assert (field in fields)

        dictionary = collections.defaultdict(list)
        key_fields = {k: fields[k] for k in key_fields}
        value_fields = {k: fields[k] for k in fields.keys() if k not in key_fields}
        key_examples = []
        key_dict = dict()
        value_examples = []

        for i, e in enumerate(examples):
            key_example = Example.fromdict({k: getattr(e, k) for k in key_fields})
            value_example = Example.fromdict({v: getattr(e, v) for v in value_fields})
            if key_example not in key_dict:
                key_dict[key_example] = len(key_examples)
                key_examples.append(key_example)

            value_examples.append(value_example)
            dictionary[key_dict[key_example]].append(i)

        self.key_dataset = Dataset(key_examples, key_fields)
        self.value_dataset = ValueDataset(value_examples, value_fields, dictionary)
        super(DictionaryDataset, self).__init__(examples, fields)

    def collate_fnc(self):
        def collate(batch):
            key_batch, value_batch = list(zip(*batch))
            key_tensors = self.key_dataset.collate_fnc()(key_batch)
            value_tensors = self.value_dataset.collate_fnc()(value_batch)
            return key_tensors, value_tensors
        return collate

    def __getitem__(self, i):
        return self.key_dataset[i], self.value_dataset[i]

    def __len__(self):
        return len(self.key_dataset)


def unique(sequence):
    seen = set()
    if isinstance(sequence[0], list):
        return [x for x in sequence if not (tuple(x) in seen or seen.add(tuple(x)))]
    else:
        return [x for x in sequence if not (x in seen or seen.add(x))]


class PairedDataset(Dataset):
    def __init__(self, examples, fields):
        assert ('image' in fields)
        assert ('text' in fields)
        super(PairedDataset, self).__init__(examples, fields)
        self.image_field = self.fields['image']
        self.text_field = self.fields['text']

    def image_set(self):
        img_list = [e.image for e in self.examples]
        image_set = unique(img_list)
        examples = [Example.fromdict({'image': i}) for i in image_set]
        dataset = Dataset(examples, {'image': self.image_field})
        return dataset

    def text_set(self):
        text_list = [e.text for e in self.examples]
        text_list = unique(text_list)
        examples = [Example.fromdict({'text': t}) for t in text_list]
        dataset = Dataset(examples, {'text': self.text_field})
        return dataset

    def image_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        dataset = DictionaryDataset(self.examples, fields, key_fields='image')
        return dataset

    def text_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        dataset = DictionaryDataset(self.examples, fields, key_fields='text')
        return dataset

    @property
    def splits(self):
        raise NotImplementedError


class COCO(PairedDataset):
    def __init__(self, image_field, text_field,train_df,test_df,val_df):
        
        train_image = train_df['image'].values
        train_caption = train_df['caption'].values
        train_input_img1 = train_df['input_img1'].values
        train_input_img2 = train_df['input_img2'].values
        train_input_img3 = train_df['input_img3'].values
        
        test_image = test_df['image'].values
        test_caption = test_df['caption'].values
        test_input_img1 = test_df['input_img1'].values
        test_input_img2 = test_df['input_img2'].values
        test_input_img3 = test_df['input_img3'].values
   
        val_image = val_df['image'].values
        val_caption = val_df['caption'].values
        val_input_img1 = val_df['input_img1'].values
        val_input_img2 = val_df['input_img2'].values
        val_input_img3 = val_df['input_img3'].values

        self.train_examples, self.val_examples, self.test_examples = self.get_samples(train_image, train_caption, train_input_img1,train_input_img2,train_input_img3,
                test_image, test_caption, test_input_img1,test_input_img2,test_input_img3,
                val_image, val_caption, val_input_img1,val_input_img2,val_input_img3)

        examples = self.train_examples + self.val_examples + self.test_examples
        super(COCO, self).__init__(examples, {'image': image_field, 'text': text_field})

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields)
        val_split = PairedDataset(self.val_examples, self.fields)
        test_split = PairedDataset(self.test_examples, self.fields)
        return train_split, val_split, test_split

    @classmethod
    def get_samples(self,
                train_image, train_caption, train_input_img1,train_input_img2,train_input_img3,
                test_image, test_caption, test_input_img1,test_input_img2,test_input_img3,
                val_image, val_caption, val_input_img1,val_input_img2,val_input_img3):
        train_samples = []
        val_samples = []
        test_samples = []

        for i in range(len(train_image)):
            img_path = (train_image[i])+'+'+(train_input_img1[i])+'+'+(train_input_img2[i])+'+'+(train_input_img3[i])
            img_path = str(img_path)

            example = Example.fromdict({'image': img_path, 'text': str(train_caption[i])})
            train_samples.append(example)

        for i in range(len(test_image)):
            img_path = (test_image[i])+'+'+(test_input_img1[i])+'+'+(test_input_img2[i])+'+'+(test_input_img3[i])
            img_path = str(img_path)

            example = Example.fromdict({'image': img_path, 'text': str(test_caption[i])})
            test_samples.append(example)

        for i in range(len(val_image)):
            img_path = (val_image[i])+'+'+(val_input_img1[i])+'+'+(val_input_img2[i])+'+'+(val_input_img3[i])
            img_path = str(img_path)

            example = Example.fromdict({'image': img_path, 'text': str(val_caption[i])})
            val_samples.append(example)

        return train_samples, val_samples, test_samples
