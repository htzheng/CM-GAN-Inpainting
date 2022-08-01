import os
import json
import numpy as np
import PIL.Image
import random
import cv2
import pyspng
from scipy import ndimage
from dnnlib.util import EasyDict
from mask_comod_generator import RandomMask

def make_dataset(dir, recursive=False, read_cache=False, write_cache=False):
    IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.webp']
    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def make_dataset_rec(dir, images):
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, dnames, fnames in sorted(os.walk(dir, followlinks=True)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    images = []
    if read_cache:
        possible_filelist = os.path.join(dir, 'files.list')
        if os.path.isfile(possible_filelist):
            with open(possible_filelist, 'r') as f:
                images = f.read().splitlines()
                return images
    if recursive:
        make_dataset_rec(dir, images)
    else:
        assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir
        for root, dnames, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    if write_cache:
        filelist_cache = os.path.join(dir, 'files.list')
        with open(filelist_cache, 'w') as f:
            for path in images:
                f.write("%s\n" % path)
            print('wrote filelist cache at %s' % filelist_cache)
    return images


class OTMask:
    '''
        object-aware mask generator
            mask_mode='mix_exclude_fg_new' is the default object-aware mask
    '''
    def __init__(self, panoptic_metadata, object_mask_path):
        ## some misc config
        self._ocr = True
        self._load_segmentation = True
        self._type = 'dir'
        self._resolution = 512
        self._crop_mode = 'random_crop'
        self._panoptic = True
        self._deterministic_mask = False
        self._mask_mode = 'mix_exclude_fg_new'   # the default masking mode

        ## get object mask file list
        self._object_mask_path = object_mask_path
        self._obj_mask_list = make_dataset(self._object_mask_path, recursive=True, read_cache=False, write_cache=False)

        ## CoModGAN mask generator
        self._mask_comod_generator = RandomMask(self._resolution, hole_range=[0., 1.]) # haitian: set mask generator

        ## the class label of the panoptic annotation can be loaded from the metadata
        if self._panoptic:
            try:
                # detectron2 can be install with:  python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'  (from https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
                from detectron2.data import MetadataCatalog 
                self._panoptic_metadata = MetadataCatalog.get('coco_2017_val_panoptic_separated')
                stuff_classes = self._panoptic_metadata.stuff_classes
            except:
                with open('_panoptic_metadata.txt', 'r') as f:
                    stuff_classes = json.load(f)

    def _open_file(self, fname, is_label=False, is_mask=False):
        if self._type == 'dir':
            if is_label:
                assert os.path.isfile(fname), f"file {fname} not exist"
                return open(fname, 'rb')
            if is_mask:
                assert os.path.isfile(fname), f"file {fname} not exist"
                return open(fname, 'rb')
            # fname = os.path.join(self._path, fname)
            assert os.path.isfile(fname), f"file {fname} not exist"
            return open(fname, 'rb')
        if self._type == 'zip':
            # raise NotImplementedError
            return self._get_zipfile().open(fname, 'r')
        return None


    def _load_object_mask(self):
        obj_mask_cfg = EasyDict()
        obj_mask_cfg.object_num = np.random.choice([1,2], p=(0.5, 0.5)) #2
        obj_mask_cfg.dilation = np.random.randint(0, 8)
        obj_mask_cfg.apply_flip = np.random.choice([True, False], p=(0.5, 0.5))
        obj_mask_cfg.apply_scaling = np.random.choice([True, False], p=(0.8, 0.2))
        obj_mask_cfg.scaling = np.random.uniform(0.75, 1.25)
        obj_mask_cfg.apply_jitter = np.random.choice([True, False], p=(0.8, 0.2))
        obj_mask_cfg.jitter_x = np.random.randint(-128-64, 128+64)
        obj_mask_cfg.jitter_y = np.random.randint(-128-64, 128+64)

        obj_mask_merged = np.zeros((self._resolution, self._resolution), dtype=np.uint8)
        for _ in range(obj_mask_cfg.object_num):
            obj_fname = self._obj_mask_list[np.random.randint(0, len(self._obj_mask_list))]
            with open(os.path.join(obj_fname), 'rb') as f:
                obj_mask = pyspng.load(f.read())
                obj_mask = (obj_mask!=0).astype(np.uint8)
                obj_mask = cv2.resize(obj_mask, dsize=(self._resolution, self._resolution), interpolation=cv2.INTER_NEAREST)
                if obj_mask.ndim==3:
                    obj_mask = obj_mask[:,:,0]
    
                ## augmentation
                # flip
                if obj_mask_cfg.apply_flip:
                    obj_mask = obj_mask[:,::-1]
                # dilation
                struct = ndimage.generate_binary_structure(2, 1)
                obj_mask = ndimage.binary_dilation(obj_mask, structure=struct, iterations=20) # obj_mask_cfg.dilation
                obj_mask = obj_mask.astype(np.uint8)

                # jitter
                if obj_mask_cfg.apply_jitter:
                    obj_mask = np.roll(obj_mask, obj_mask_cfg.jitter_y, axis=0)
                    obj_mask = np.roll(obj_mask, obj_mask_cfg.jitter_x, axis=1)


                obj_mask_merged = obj_mask + obj_mask_merged

            obj_mask_merged = (obj_mask_merged>=1).astype(np.uint8)

        return obj_mask_merged


    def _load_raw_image_label(self, fname, fname_json, crop_config=None, panoptic=False):

        with self._open_file(fname, is_label=True) as f:
            ## read
            label = PIL.Image.open(f)
            H,W = label.size
            # if pyspng is not None and self._file_ext(fname) == '.png':
            #     label = pyspng.load(f.read())
            # else:
            #     label = np.array(PIL.Image.open(f))
            ## resize
            label = self.resize_by_shorter_side(label, nearest=True)
            label = np.array(label)

            ## crop
            label, crop_config = self.get_crop_pos_and_crop(label, crop_config)


        label = label[np.newaxis, :, :]
        
        # label json annotation
        if panoptic:
            with self._open_file(fname_json, is_label=True) as f:
                anno = json.load(f)
        else:
            anno = None

        return label, anno

    def resize_by_shorter_side(self, image, nearest=False, resize_mode='direct_resize'):
        H,W = image.size
        if H==self._resolution and W==self._resolution:
            return image

        if resize_mode=='direct_resize':
            H_new, W_new = self._resolution, self._resolution

        image = image.resize((H_new, W_new), PIL.Image.NEAREST if nearest else PIL.Image.LANCZOS)
        return image

    def get_crop_pos_and_crop(self, image, crop_config):
        if crop_config is None:
            if self._crop_mode=='center_crop':
                crop_y_pos = (image.shape[0]-self._resolution)//2
                crop_x_pos = (image.shape[1]-self._resolution)//2
            elif self._crop_mode=='random_crop':
                crop_y_pos = np.random.randint(0, image.shape[0]-self._resolution+1)
                crop_x_pos = np.random.randint(0, image.shape[1]-self._resolution+1)
            crop_config = {'crop_y_pos':crop_y_pos, 'crop_x_pos':crop_x_pos}
        
        image_crop = image[crop_config['crop_y_pos']:crop_config['crop_y_pos']+self._resolution, crop_config['crop_x_pos']:crop_config['crop_x_pos']+self._resolution, ...]
        # if image_crop.shape[0]!=self._resolution or image_crop.shape[1]!=self._resolution:
        #     print(image_crop.shape)
        return image_crop, crop_config            

    def _load_raw_image(self, fname, crop_config=None):
        with self._open_file(fname) as f:
            ## read
            image = PIL.Image.open(f).convert("RGB")
            H,W = image.size
            # if pyspng is not None and self._file_ext(fname) == '.png':
            #     image = pyspng.load(f.read())
            # else:
            #     # image = np.array(PIL.Image.open(f).convert("RGB").resize((self._resolution, self._resolution), resample=PIL.Image.BICUBIC)) # haitian: do resizing

            ## resize
            image = self.resize_by_shorter_side(image, nearest=False)

            image = np.array(image)
            ## crop
            image, crop_config = self.get_crop_pos_and_crop(image, crop_config)
            crop_config['size'] = (H,W)

        # if image.ndim == 2:
        #     image = image[:, :, np.newaxis] # HW => HWC
        assert image.ndim == 3
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image, crop_config, fname

    def _get_fg_overlapped(self, image_label, image_label_anno, comod_mask, ratio=0.5):
        if self._panoptic:
            fg_list = [item['id'] for item in image_label_anno if item['isthing']]

        image_label_inside = image_label[0,:,:]*(1-comod_mask) + (image_label[0,:,:]*0 + 255)*comod_mask
        labels, counts = np.unique(image_label[0,:,:], return_counts=True)
        label_count = {int(l):float(c) for (l,c) in zip(labels, counts) if (l in fg_list)}

        labels_inside, counts_inside = np.unique(image_label_inside, return_counts=True)
        label_count_inside = {int(l):float(c) for (l,c) in zip(labels_inside, counts_inside) if (l in fg_list)}
        label_count_inside.update({int(l):0. for l in labels if l not in label_count_inside})

        label_ratio_inside = {int(l): label_count_inside[l]/label_count[l] for l in label_count}
        label_to_exclude = [int(l) for l in label_ratio_inside if (label_ratio_inside[l]>ratio)]

        fg_to_exclude = np.isin(image_label[0,:,:], label_to_exclude)
        

        fg_to_exclude = np.logical_not(np.isin(image_label[0,:,:], label_to_exclude))
        fg_to_exclude = fg_to_exclude.astype(np.uint8)
        fg_to_exclude = 1 - fg_to_exclude

        random_dilation = np.random.randint(2,5) # dilate the object mask so that masked image doesn't leak background information at the boundary of objects
        if random_dilation!=0:
            struct = ndimage.generate_binary_structure(2, 1)
            fg_to_exclude = ndimage.binary_erosion(fg_to_exclude, structure=struct, iterations=random_dilation)

        return fg_to_exclude


    def _load_raw_mask(self, idx, image_label=None, image_label_anno=None, deterministic_mask=False):
        seed = idx+10 if deterministic_mask else None
        if self._mask_mode=='mix_exclude_fg_new':
            mask_mode_list = ['object', 'comod', 'rectangle_only']
            mask_mode = random.choices(mask_mode_list, weights=[45,45,10])[0]

            ratio = 0.5 # the 50% rule
            if mask_mode=='comod':
                mask = self._mask_comod_generator(seed).astype(np.uint8)
            elif mask_mode=='object':
                mask = 1-self._load_object_mask()
            elif mask_mode=='rectangle_only':
                mask = self._mask_comod_generator.call_rectangle(seed).astype(np.uint8)
            
            fg_overlapped = self._get_fg_overlapped(image_label, image_label_anno, mask, ratio=ratio)
            comod_exclude_fg = (fg_overlapped + mask).clip(0,1)
            return comod_exclude_fg
        else:
            raise NotImplementedError

    def get_mask(self, idx, image_label=None, image_label_anno=None):
        mask = self._load_raw_mask(idx, image_label=image_label, image_label_anno=image_label_anno, deterministic_mask=self._deterministic_mask)
        return mask[None,:,:]

    def __call__(self, image_fname, anno_seg_fname, anno_json_fname, max_trail=6, mask_accept_threshold=0.95):
        ''' 
            the function for generating random object-aware masks given the image and panoptic annotation '''
        trial = 0
        while True:
            image_, crop_config, image_name = self._load_raw_image(image_fname, crop_config=None)
            image_label_, image_label_anno_ = self._load_raw_image_label(anno_seg_fname, anno_json_fname, crop_config=crop_config, panoptic=self._panoptic)
            mask_ = self.get_mask(idx=0, image_label=image_label_, image_label_anno=image_label_anno_)
            
            if (mask_.mean() <= mask_accept_threshold or trial>max_trail) or \
                    self._load_fixed_mask:  # if load fixed mask, assume that we don't have to corp image and generate mask randomly
                break
            trial += 1
        masked_image = image_ * mask_
        return image_, mask_[0,:,:], masked_image
        
if __name__=="__main__":
    image_fname = 'Places365_test_00000002.jpg'
    anno_seg_fname = 'Places365_test_00000002.png'
    anno_json_fname = 'Places365_test_00000002.json'
    
    mask_fname = 'output_mask.png'
    masked_image_fname = 'output_masked_image.png'

    mask_generator = OTMask(panoptic_metadata='_panoptic_metadata.txt', object_mask_path='./object_masks/')
    image, mask, masked_image = mask_generator(image_fname, anno_seg_fname, anno_json_fname)
    
    # display
    image = PIL.Image.fromarray(image.transpose(1,2,0))
    image.show()
    masked_image = PIL.Image.fromarray(masked_image.transpose(1,2,0))
    masked_image.show()
    mask = PIL.Image.fromarray(mask*255)
    mask.show()

    masked_image.save(masked_image_fname)
    mask.save(mask_fname)
