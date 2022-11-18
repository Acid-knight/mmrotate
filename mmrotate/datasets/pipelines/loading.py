# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmdet.datasets.pipelines import LoadImageFromFile

from ..builder import ROTATED_PIPELINES


@ROTATED_PIPELINES.register_module()
class LoadPatchFromImage(LoadImageFromFile):
    """Load an patch from the huge image.

    Similar with :obj:`LoadImageFromFile`, but only reserve a patch of
    ``results['img']`` according to ``results['win']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with image in ``results['img']``.

        Returns:
            dict: The dict contains the loaded patch and meta information.
        """

        img = results['img']
        x_start, y_start, x_stop, y_stop = results['win']
        width = x_stop - x_start
        height = y_stop - y_start

        patch = img[y_start:y_stop, x_start:x_stop]
        if height > patch.shape[0] or width > patch.shape[1]:
            patch = mmcv.impad(patch, shape=(height, width))

        if self.to_float32:
            patch = patch.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = patch  # 这里重写了原始的['img']键,因为多了一个裁剪操作
        results['img_shape'] = patch.shape
        results['ori_shape'] = patch.shape
        results['img_fields'] = ['img']
        return results


import os.path as osp
from PIL import Image
import numpy as np
import torch

@ROTATED_PIPELINES.register_module()
class LoadImagewithMask(LoadImageFromFile):
    """Load an patch from the huge image.

    Similar with :obj:`LoadImageFromFile`, but only reserve a patch of
    ``results['img']`` according to ``results['win']``.
    """
    def get_img_mask(self,filename,ori_filename):
        # get road prior feature mask of imgs and do the same data argu
        # img_metas:list(dict{10}:(ori_shape,img_shape,pad_shape,scale_factor,flip,flip_direction,img_norm_cfg,batch_input_shape)

        # mask=[]
        root = filename  # 总路径
        p_root=root.split('/')[:-1]
        pa_root=''
        for s in p_root:
            pa_root=osp.join(pa_root,s)

        mask_image = Image.open(pa_root + '_mask/' + ori_filename.split('.')[0]+'_mask.png')  # 读取images_mask文件夹里的mask
        # 读入时是三通道图像

        mask_image=mask_image.convert('L')  # 转换为单通道
        mask_image=np.array(mask_image)
        return mask_image


    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order)
        if self.to_float32:
            img = img.astype(np.float32) # ori img

        ori_filename=results['img_info']['filename']
        # 读取先验掩膜
        mask = self.get_img_mask(filename,ori_filename)
        mask=np.expand_dims(mask,axis=2) # 扩充维度

        # 交换维度从(H,W,C)变为(C,H,W)
        img=np.transpose(img,(2,0,1))
        mask = np.transpose(mask,(2, 0, 1))
        # 在C维度拼接
        # img=np.stack((img,mask),axis=0)
        img = np.concatenate((img, mask), axis=0)
        # 交换回去(C,H,W)->(H,W,C)
        img=np.transpose(img,(1,2,0))

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        # results['mask'] = mask  # 保存mask信息
        return results

