#
#   Lightnet dataset that works with brambox annotations
#   Copyright EAVISE
#
# https://eavise.gitlab.io/lightnet/_modules/lightnet/models/_dataset_brambox.html#BramboxDataset
# https://eavise.gitlab.io/brambox/notes/02-getting_started.html#Loading-data

import os
import copy
import logging
from PIL import Image
import numpy as np
import lightnet.data as lnd
from pathflowai.utils import load_sql_df
import dask.array as da
from os.path import join

try:
    import brambox as bb
except ImportError:
    bb = None

__all__ = ['BramboxDataset']
log = logging.getLogger(__name__)

# ADD IMAGE ANNOTATION TRANSFORM
# ADD TRAIN VAL TEST INFO
class BramboxPathFlowDataset(lnd.Dataset):
    """ Dataset for any brambox annotations.

    Args:
        annotations (dataframe): Dataframe containing brambox annotations
        input_dimension (tuple): (width,height) tuple with default dimensions of the network
        class_label_map (list): List of class_labels
        identify (function, optional): Lambda/function to get image based of annotation filename or image id; Default **replace/add .png extension to filename/id**
        img_transform (torchvision.transforms.Compose): Transforms to perform on the images
        anno_transform (torchvision.transforms.Compose): Transforms to perform on the annotations

    Note:
        This dataset opens images with the Pillow library
    """
    def __init__(self, input_dir, patch_info_file, patch_size, annotations, input_dimension, class_label_map=None, identify=None, img_transform=None, anno_transform=None):
        if bb is None:
            raise ImportError('Brambox needs to be installed to use this dataset')
        super().__init__(input_dimension)

        self.annos = annotations
        self.annos['ignore']=0
        self.annos['class_label']=self.annos['class_label'].astype(int)#-1
        print(self.annos['class_label'].unique())
        #print(self.annos.shape)
        self.keys = self.annos.image.cat.categories # stores unique patches
        #print(self.keys)
        self.img_tf = img_transform
        self.anno_tf = anno_transform
        self.patch_info=load_sql_df(patch_info_file, patch_size)
        IDs=self.patch_info['ID'].unique()
        self.slides = {slide:da.from_zarr(join(input_dir,'{}.zarr'.format(slide))) for slide in IDs}
        self.id = lambda k: k.split('/')

        # Add class_ids
        if class_label_map is None:
            log.warning(f'No class_label_map given, generating it by sorting unique class labels from data alphabetically, which is not always deterministic behaviour')
            class_label_map = list(np.sort(self.annos.class_label.unique()))
        self.annos['class_id'] = self.annos.class_label.map(dict((l, i) for i, l in enumerate(class_label_map)))

    def __len__(self):
        return len(self.keys)

    @lnd.Dataset.resize_getitem
    def __getitem__(self, index):
        """ Get transformed image and annotations based of the index of ``self.keys``

        Args:
            index (int): index of the ``self.keys`` list containing all the image identifiers of the dataset.

        Returns:
            tuple: (transformed image, list of transformed brambox boxes)
        """
        if index >= len(self):
            raise IndexError(f'list index out of range [{index}/{len(self)-1}]')

        # Load
        #print(self.keys[index])
        ID,x,y,patch_size=self.id(self.keys[index])
        x,y,patch_size=int(x),int(y),int(patch_size)
        img = self.slides[ID][x:x+patch_size,y:y+patch_size].compute()#Image.open(self.id(self.keys[index]))
        anno = bb.util.select_images(self.annos, [self.keys[index]])

        # Transform
        if self.img_tf is not None:
            img = self.img_tf(img)
        if self.anno_tf is not None:
            anno = self.anno_tf(anno)

        return img, anno
