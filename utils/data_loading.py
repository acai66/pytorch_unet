import logging
import math
from os import listdir
from os.path import splitext
from pathlib import Path
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import imgaug as ia
from imgaug import augmenters as iaa
from tqdm import tqdm


NUM_THREADS = 8




class BasicDataset(Dataset):

    
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '', cache_images=False, low_aug=True, square=True, square_width=192):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.low_aug = low_aug
        self.square = square
        self.square_width = square_width
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        # self.ids = self.ids[:2000]    # For faster debug.
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        
        self.cache_images = cache_images
        if self.cache_images:
            logging.info(f'Caching datasets... ')
            num = len(self.ids)
            self.imgs, self.masks = [None] * num, [None] * num
            
            for i in tqdm(range(len(self.ids))):
                mask_file = list(self.masks_dir.glob(self.ids[i] + self.mask_suffix + '.*'))
                img_file = list(self.images_dir.glob(self.ids[i] + '.*'))

                assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {self.ids[i]}: {mask_file}'
                assert len(img_file) == 1, f'Either no image or multiple images found for the ID {self.ids[i]}: {img_file}'
                self.imgs[i] = self.load(img_file[0])
                self.masks[i] = self.load(mask_file[0])
            
            '''
            with Pool(NUM_THREADS) as pool:
                def load_image_warpper(self, i):
                    return 1
                results = pool.imap(load_image, zip(repeat(self), range(num)))
                pbar = tqdm(results, total=num)
                for x in pbar:
                    i, img, mask = x
                    self.imgs[i], self.masks[i] = img, mask
                    pbar.desc = f'Caching images...'
            pbar.close()
            '''
                

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray
    
    def preprocess_all(self, img, mask, scale):
        w, h = img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = img.resize((newW, newH))
        pil_mask = mask.resize((newW, newH), resample=Image.NEAREST)
        img_nd = np.array(pil_img)
        mask_nd = np.array(pil_mask)
        seq = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.Crop(px=(0, 16))),
            iaa.Affine(rotate=(-90, 90)),
            iaa.Sometimes(0.5, iaa.Fliplr(0.5)),
            iaa.Sometimes(0.5, iaa.GaussianBlur((0, 0.5)),
                iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
                random_state=True)
        ]) if self.low_aug else iaa.Sequential([
            iaa.Sometimes(0.5, iaa.Fliplr(0.3))
        ])
        seg_map = ia.SegmentationMapsOnImage(mask_nd, shape = img_nd.shape)
        image_aug, seg_aug = seq(image=img_nd, segmentation_maps = seg_map)
        seg_map = seg_aug.get_arr()
        img_trans = image_aug.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        # seg_map = np.expand_dims(seg_map, axis=2)
        seg_trans = seg_map # .transpose((2, 0, 1))
        # if seg_trans.max() > 1:
            # seg_trans = seg_trans / 255

        return img_trans, seg_trans
        

    # @classmethod
    def load(self, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            ret = Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            ret = Image.fromarray(torch.load(filename).numpy())
        else:
            ret = Image.open(filename).convert("RGB")
        if self.square:
            ret = ret.resize((self.square_width, self.square_width), Image.NEAREST)
        if ret.size[0] < 64:
            ret = ret.resize((64, math.ceil(ret.size[1] * 64 / ret.size[0])), Image.NEAREST)
        if ret.size[1] < 64:
            ret = ret.resize((math.ceil(ret.size[0] * 64 / ret.size[1]), 64), Image.NEAREST)
        return ret

    def __getitem__(self, idx):
        
        name = self.ids[idx]
        if self.cache_images:
            mask = self.masks[idx]
            img = self.imgs[idx]
        else:
            mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
            img_file = list(self.images_dir.glob(name + '.*'))

            assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
            assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
            mask = self.load(mask_file[0])
            img = self.load(img_file[0])

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        # img = self.preprocess(img, self.scale, is_mask=False)
        # mask = self.preprocess(mask, self.scale, is_mask=True)
        img, mask = self.preprocess_all(img, mask, self.scale)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(dummy):
    
    self, i = dummy
    
    mask_file = list(self.masks_dir.glob(self.ids[i] + self.mask_suffix + '.*'))
    img_file = list(self.images_dir.glob(self.ids[i] + '.*'))

    assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {self.ids[i]}: {mask_file}'
    assert len(img_file) == 1, f'Either no image or multiple images found for the ID {self.ids[i]}: {img_file}'
    img = self.load(img_file[0])
    mask = self.load(mask_file[0])
    
    return i, img, mask


class FishDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1, low_aug=False):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='', low_aug=low_aug, square=True, square_width=320)
