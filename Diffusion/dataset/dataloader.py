from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
    RandZoomd,
    RandCropByLabelClassesd,
)

import collections.abc
import math
import pickle
import shutil
import sys
import tempfile
import threading
import time
import warnings
from copy import copy, deepcopy
import h5py, os

from monai.transforms import Lambda

import numpy as np
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

sys.path.append("..") 

from torch.utils.data import Subset

from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.io.array import LoadImage, SaveImage
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
DEFAULT_POST_FIX = PostFix.meta()

class UniformDataset(Dataset):
    def __init__(self, data, transform, datasetkey):
        super().__init__(data=data, transform=transform)
        self.dataset_split(data, datasetkey)
        self.datasetkey = datasetkey
    
    def dataset_split(self, data, datasetkey):
        self.data_dic = {}
        for key in datasetkey:
            self.data_dic[key] = []
        for img in data:
            key = get_key(img['name'])
            self.data_dic[key].append(img)
        
        self.datasetnum = []
        for key, item in self.data_dic.items():
            assert len(item) != 0, f'the dataset {key} has no data'
            self.datasetnum.append(len(item))
        self.datasetlen = len(datasetkey)
    
    def _transform(self, set_key, data_index):
        data_i = self.data_dic[set_key][data_index]
        return apply_transform(self.transform, data_i) if self.transform is not None else data_i
    
    def __getitem__(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        set_key = self.datasetkey[set_index]
        # data_index = int(index / self.__len__() * self.datasetnum[set_index])
        data_index = np.random.randint(self.datasetnum[set_index], size=1)[0]
        return self._transform(set_key, data_index)


class UniformCacheDataset(CacheDataset):
    def __init__(self, data, transform, cache_rate, datasetkey):
        super().__init__(data=data, transform=transform, cache_rate=cache_rate)
        self.datasetkey = datasetkey
        self.data_statis()
    
    def data_statis(self):
        data_num_dic = {}
        for key in self.datasetkey:
            data_num_dic[key] = 0
        # breakpoint()
        for img in self.data:
            key = get_key(img['name'])
            data_num_dic[key] += 1

        self.data_num = []
        for key, item in data_num_dic.items():
            assert item != 0, f'the dataset {key} has no data'
            self.data_num.append(item)
        
        self.datasetlen = len(self.datasetkey)
    
    def index_uniform(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        data_index = np.random.randint(self.data_num[set_index], size=1)[0]
        post_index = int(sum(self.data_num[:set_index]) + data_index)
        return post_index

    def __getitem__(self, index):
        post_index = self.index_uniform(index)
        # print(post_index, self.__len__())
        return self._transform(post_index)

class LoadImageh5d(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


    def register(self, reader: ImageReader):
        self._loader.register(reader)


    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key], reader)
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]
        return d


class RandZoomd_select(RandZoomd):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = get_key(name)
        if (key not in ['10_03', '10_06', '10_07', '10_08', '10_09', '10_10']):
            return d
        d = super().__call__(d)
        return d


class RandCropByPosNegLabeld_select(RandCropByPosNegLabeld):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = get_key(name)
        if key in ['10_03', '10_07', '10_08', '04', '05']: # if key in ['10_03', '10_07', '10_08', '04']
            return d
        d = super().__call__(d)
        return d

class RandCropByLabelClassesd_select(RandCropByLabelClassesd):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = get_key(name)
        # print('key',key)
        if key not in ['10_03', '10_07', '10_08', '04', '05']: # if key in ['10_03', '10_07', '10_08', '04']
            return d
        d = super().__call__(d)
        return d

class Compose_Select(Compose):
    def __call__(self, input_):
        name = input_['name']
        key = get_key(name)
        for index, _transform in enumerate(self.transforms):
            # for RandCropByPosNegLabeld and RandCropByLabelClassesd case
            if (key in ['10_03', '10_07', '10_08', '04']) and (index == 8):
                continue
            elif (key not in ['10_03', '10_07', '10_08', '04']) and (index == 9):
                continue
            # for RandZoomd case
            if (key not in ['10_03', '10_06', '10_07', '10_08', '10_09', '10_10']) and (index == 7):
                continue
            input_ = apply_transform(_transform, input_, self.map_items, self.unpack_items, self.log_stats)
        return input_

from itertools import cycle
current_index = 0
def choose_different_value(x, num_samples):
    global current_index
    num_texts = len(x["text"])
    if current_index >= num_samples:
        current_index = 0
    current_choice = x["text"][current_index]
    current_index = (current_index + 1) % num_texts
    return {**x, "text": current_choice}

import random
def get_loader(args):
    def check_shapes(stage):
        def _check_shapes(data):
            image = data['image']
            label = data['label']
            return data
        return _check_shapes

    train_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "label"]), 
            Lambda(check_shapes("After LoadImageh5d")),
            AddChanneld(keys=["image", "label"]),
            Lambda(check_shapes("After AddChanneld")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Lambda(check_shapes("After Orientationd")),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
                align_corners=True,
            ),
            Lambda(check_shapes("After Spacingd")),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
            Lambda(check_shapes("After SpatialPadd")),
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                ratios=[0, 1, 1],
                num_classes=3,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=-1,
            ),
            Lambda(check_shapes("After RandCropByLabelClassesd")),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            Lambda(check_shapes("After RandRotate90d")),
            Lambda(lambda x: choose_different_value(x, args.num_samples)),
            ToTensord(keys=["image", "label"]),
        ]
    )


    val_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                ratios=[0, 0, 1],
                num_classes=3,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ),
            Lambda(lambda x: choose_different_value(x,args.num_samples)),
            ToTensord(keys=["image", "label"]),
        ]
    )

    def load_data(file_path, root_path, label_root_path):
        img_list = []
        lbl_list = []
        text_list = []
        name_list = []

        with open(file_path) as f:
            for line in f:
                parts = line.strip().split(maxsplit=2)  
                name = parts[1].split('.')[0]
                img_list.append(root_path + parts[0])
                lbl_list.append(label_root_path + parts[1])

                text_descriptions = parts[2].split(' ' * 10)  
                text_descriptions = [desc.strip() for desc in text_descriptions if desc.strip()] 

                text_list.append(text_descriptions)
                name_list.append(name)

        return img_list, lbl_list, text_list, name_list

    if args.phase == 'train':
        train_img, train_lbl, train_text, train_name = [], [], [], []

        for item in args.dataset_list:
            img, lbl, text, name = load_data(os.path.join(args.data_txt_path, item, f'real_tumor.txt'),
                                             args.data_root_path, args.label_root_path)
            train_img.extend(img)
            train_lbl.extend(lbl)
            train_text.extend(text)
            train_name.extend(name)
        print("a",train_img)
        print("b",train_lbl)
        print("c",train_text)
        print("d",train_name)
        data_dicts_train = [
            {'image': image, 'label': label, 'name': name, 'text': text}
            for image, label, name, text in zip(train_img, train_lbl, train_name, train_text)
        ]
        
        print('train len {}'.format(len(data_dicts_train)))

        if args.cache_dataset:
            if args.uniform_sample:
                train_dataset = UniformCacheDataset(data=data_dicts_train, transform=train_transforms, cache_rate=args.cache_rate, datasetkey=args.datasetkey)
            else:
                train_dataset = CacheDataset(data=data_dicts_train, transform=train_transforms, cache_rate=args.cache_rate)
        else:
            if args.uniform_sample:
                train_dataset = UniformDataset(data=data_dicts_train, transform=train_transforms, datasetkey=args.datasetkey)
            else:
                train_dataset = Dataset(data=data_dicts_train, transform=train_transforms)
        
        train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True, shuffle=True) if args.dist else None
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, 
                                  collate_fn=list_data_collate, sampler=train_sampler)
        return train_loader, train_sampler, len(train_dataset)
    
    if args.phase == 'validation':
        val_img, val_lbl, val_text, val_name = [], [], [], []
        
        for item in args.dataset_list:
            img, lbl, text, name = load_data(os.path.join(args.data_txt_path, item, 'real_huge_train_0.txt'),
                                             args.data_root_path, args.label_root_path)
            val_img.extend(img)
            val_lbl.extend(lbl)
            val_text.extend(text)
            val_name.extend(name)
        
        data_dicts_val = [
            {'image': image, 'label': label, 'name': name, 'text': text}
            for image, label, name, text in zip(val_img, val_lbl, val_name, val_text)
        ]
        
        print('val len {}'.format(len(data_dicts_val)))

        if args.cache_dataset:
            val_dataset = CacheDataset(data=data_dicts_val, transform=val_transforms, cache_rate=args.cache_rate)
        else:
            val_dataset = Dataset(data=data_dicts_val, transform=val_transforms)
        
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
        return val_loader, val_transforms, len(val_dataset)


def get_key(name):
    dataset_index = int(name[0:2])
    if dataset_index == 10:
        template_key = name[0:2] + '_' + name[17:19]
    else:
        template_key = name[0:2]
    return template_key

if __name__ == "__main__":
    train_loader, test_loader = partial_label_dataloader()
    for index, item in enumerate(test_loader):
        print(item['image'].shape, item['label'].shape, item['task_id'])
        input()