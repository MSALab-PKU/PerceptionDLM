import os
from io import BytesIO
import base64
import json
import pickle
from PIL import Image
import pyarrow.parquet as pq
import numpy as np
from typing import List, Tuple, Union, Iterable,BinaryIO
import random
import copy
import io
import tqdm
import pycocotools.mask as mask_util
import time
from .utils import alphanum_path_key
from .registry import register_dataset

# class TextDataBase(DataBase):
#     task = 'text'
#     get_fns = ['get_text']

#     def get_text(self, record) -> dict:
#         raise NotImplementedError


class ImageVQADataBase:
    # conversation = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": "https://www.ilankelman.org/stopsigns/australia.jpg"},
    #             {"type": "text", "text": "Please describe this image in detail."},
    #         ],
    #     },
    # ]
    def __init__(self, path, image_dir='', **kwargs):
        self.path = path
        self.image_dir = image_dir
        self.data = self.read(path, **kwargs)
    
    def read(self, path, **kwargs):
        ext = os.path.splitext(path)[1]
        if ext == '.csv' or ext == '.tsv' or ext == '.txt':
            raise NotImplementedError(f"Unrecognized file ext: {ext}. ")
        elif ext == '.json':
            with open(path) as f:
                data = json.load(f)
        elif ext == '.jsonl':
            with open(path) as f:
                data = [json.loads(l.rstrip('\n')) for l in f]
        else:
            raise NotImplementedError(f"Unrecognized file ext: {ext}. ")

        return data

    def _parse_image_path(self, image_dir, image_file):

        if image_file.startswith("/"):
            image_path = image_file
        else:
            image_dir = image_dir.rstrip("/")
            if os.path.basename(image_dir) == image_file.split("/")[0]:
                image_file = "/".join(image_file.split("/")[1:])
            image_path = os.path.join(image_dir, image_file)
        return image_path

    def open_images(self,image_paths):
        images = []
        for image in image_paths:
            try:
                images.append(Image.open(image).convert('RGB'))
            except:
                continue
        return images


    def get_crop(self, image, mask):
        mask = mask.astype(bool)

        coords = np.where(mask)
        if coords[0].size == 0:
            return image

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        cropped = image.crop((x_min, y_min, x_max + 1, y_max + 1))

        return cropped

    def get_img(self, record):
        
        if "images" in record:
            image_paths = [self._parse_image_path(self.image_dir, fn) for fn in record["images"]]
        elif "image" in record:
            if isinstance(record["image"], list):
                # print(record["image"])
                image_paths = [self._parse_image_path(self.image_dir, fn) for fn in record["image"]]
            else:
                image_path = self._parse_image_path(self.image_dir, record["image"])
                if os.path.isdir(image_path):
                    image_paths = [os.path.join(image_path, fn) for fn in
                                   sorted(os.listdir(image_path), key=alphanum_path_key)]
                else:
                    image_paths = [image_path]
        elif "video" in record:
            video_path = self._parse_image_path(self.image_dir, record["video"])
            image_paths = [os.path.join(video_path, fn) for fn in sorted(os.listdir(video_path), key=alphanum_path_key)]
        else:
            image_paths = []
        if len(image_paths) > 0:
            try:
                images = self.open_images(image_paths)
            except:
                for p in image_paths:
                    os.system(f"sudo chmod a+rw {p}")
                images = self.open_images(image_paths)
            if "mask_rle" in record and len(images)>0:
                mask = mask_util.decode(record["mask_rle"])
                images.append(self.get_crop(images[0], mask))
        else:
            images = []

        return {"images": images}

    def get_conv(self, record) -> dict:
        raise NotImplementedError
    def __getitem__(self, k, **kwargs) -> dict:
        record = copy.deepcopy(self.data[k])
        record.update({'id': k, 'from': self.path})
        img_meta = self.get_img(record)
        record.update(img_meta)

        conv_meta = self.get_conv(record)
        record.update(conv_meta)

        return record

    def __len__(self):
        return len(self.data)


class ImageVQAParquetDataBase:
    # conversation = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": "https://www.ilankelman.org/stopsigns/australia.jpg"},
    #             {"type": "text", "text": "Please describe this image in detail."},
    #         ],
    #     },
    # ]
    def __init__(self, path, row_size=1, **kwargs):
        self.row_size = row_size
        self.path = path
        folder = path
        self.folder = folder
        self.file_metadata, self.total_length = self.read(folder, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'file_metadata' in state:
            new_meta = []
            for meta in state['file_metadata']:
                meta_copy = meta.copy()
                meta_copy['file_handle'] = None
                new_meta.append(meta_copy)
            state['file_metadata'] = new_meta
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if hasattr(self, 'file_metadata'):
            import pyarrow.parquet as pq
            for meta in self.file_metadata:
                if 'path' in meta:
                    meta['file_handle'] = pq.ParquetFile(meta['path'])

    def read(self, folder, **kwargs):
        file_names = os.listdir(folder)
        file_paths = []
        for file_name in file_names:
            if file_name.split(".")[-1] == "parquet":
                file_paths.append(os.path.join(folder, file_name))

        file_metadata = []
        total_length = 0
        for path in tqdm.tqdm(file_paths):
            try:
                parquet_file = pq.ParquetFile(path)
            except:
                continue
            num_rows = parquet_file.metadata.num_rows
            start_idx = total_length
            end_idx = total_length + num_rows - 1

            file_metadata.append({
                'path': path,
                'num_rows': num_rows,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'file_handle': parquet_file  # 保持文件句柄以便后续读取
            })

            total_length += num_rows

        print(f"已加载 {len(file_paths)} 个Parquet文件，总记录数: {total_length}")

        return file_metadata, total_length

    def _get_img_from_bytes(self, img_bytes):
        if isinstance(img_bytes, dict) and "bytes" in img_bytes:
            img_bytes = img_bytes["bytes"]
        image_stream = io.BytesIO(img_bytes)
        pil_image = Image.open(image_stream).convert('RGB')
        return pil_image

    def get_img(self, record):
        if "images" in record and record["images"] is not None:
            try:
                images = record["images"]
                images = [self._get_img_from_bytes(img_byte) for img_byte in images]
            except:
                images = []

        elif "image" in record and record["image"] is not None:
            try:
                images = record["image"]
                images = [self._get_img_from_bytes(img_byte) for img_byte in images]
            except:
                images = []

        else :
            images = []
        return {"images": images}

    def get_conv(self, record) -> dict:
        raise NotImplementedError

    def _get_file_info(self, global_index: int) -> Tuple[str, int, pq.ParquetFile]:
        """
        根据全局索引找到对应的文件和本地索引

        Args:
            global_index: 全局索引

        Returns:
            文件路径, 本地索引, ParquetFile对象
        """
        if global_index < 0 or global_index >= self.total_length:
            raise IndexError(f"全局索引 {global_index} 超出范围 [0, {self.total_length - 1}]")

        # 二分查找找到对应的文件
        left, right = 0, len(self.file_metadata) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.file_metadata[mid]['start_idx'] <= global_index <= self.file_metadata[mid]['end_idx']:
                file_info = self.file_metadata[mid]
                local_index = global_index - file_info['start_idx']
                return file_info['path'], local_index, file_info['file_handle']
            elif global_index < self.file_metadata[mid]['start_idx']:
                right = mid - 1
            else:
                left = mid + 1
        raise ValueError(f"无法找到索引 {global_index} 对应的文件")

    def read_single(self, global_index: int) -> dict:
        """
        读取单个全局索引对应的记录

        Args:
            global_index: 全局索引

        Returns:
            包含记录数据的字典
        """
        _, local_idx, parquet_file = self._get_file_info(global_index)

        # 获取 row group 信息
        num_row_groups = parquet_file.metadata.num_row_groups
        
        # 找到 local_idx 所在的 row group
        cumulative_rows = 0
        row_group_idx = 0
        offset_in_row_group = local_idx
        
        for rg_idx in range(num_row_groups):
            rg_num_rows = parquet_file.metadata.row_group(rg_idx).num_rows
            if cumulative_rows + rg_num_rows > local_idx:
                row_group_idx = rg_idx
                offset_in_row_group = local_idx - cumulative_rows
                break
            cumulative_rows += rg_num_rows
        
        # 读取对应的 row group，转为 pandas 以避免嵌套数据类型的 Arrow 限制
        try:
            table = parquet_file.read_row_group(row_group_idx, columns=None)
            # 转为 pandas 再取行，避免 ArrowNotImplementedError
            df = table.to_pandas()
            row = df.iloc[offset_in_row_group].to_dict()
        except Exception:
            # 回退方案：直接读取整个文件的指定行
            table = parquet_file.read()
            df = table.to_pandas()
            row = df.iloc[local_idx].to_dict()
        return row, _

    def __getitem__(self, k, **kwargs) -> dict:
        record, path = self.read_single(k)
        record.update({'id': k, 'from': path})

        img_meta = self.get_img(record)
        record.update(img_meta)

        conv_meta = self.get_conv(record)
        record.update(conv_meta)
        
        return record

    def __len__(self):
        return self.total_length