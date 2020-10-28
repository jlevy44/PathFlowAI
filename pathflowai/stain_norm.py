import cv2
import sys
import fire
import histomicstk
import histomicstk as htk
import openslide
import dask
import tqdm
import numpy as np
from dask.diagnostics import ProgressBar
from pathflowai.utils import generate_tissue_mask
from histomicstk.preprocessing.color_normalization.\
    deconvolution_based_normalization import deconvolution_based_normalization

W_target = np.array([
            [0.6185391,  0.1576997,  -0.01119131],
            [0.7012888,  0.8638838,  0.45586256],
            [0.3493163,  0.4657428, -0.85597752]
        ])

def return_norm_image(img,mask):
    img=deconvolution_based_normalization(
        img, W_source=W_source, W_target=W_target, im_target=None,
        stains=['hematoxylin', 'eosin'], mask_out=~mask,
        stain_unmixing_routine_params={"I_0":215})
    return img

def stain_norm(svs,compression=10,patch_size=1024):
    img = openslide.open_slide(svs)
    image = np.array(img.read_region((0,0), 0, img.level_dimensions[0]))[...,:3]
    mask=generate_tissue_mask(image,compression=compression,keep_holes=False)
    img_small=cv2.resize(image,None,fx=1/compression,fy=1/compression)
    mask_small=cv2.resize(mask.astype(int),None,fx=1/compression,fy=1/compression,interpolation=cv2.INTER_NEAREST).astype(bool)
    W_source = htk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(img_small, 215)
    W_source=htk.preprocessing.color_deconvolution._reorder_stains(W_source)
    res=[]
    coords=[]
    for i in np.arange(0,image.shape[0]-patch_size,patch_size):
        for j in np.arange(0,image.shape[1]-patch_size,patch_size):
            if mask[i:i+patch_size,j:j+patch_size].mean():
                coords.append((i,j))
                res.append(dask.delayed(return_norm_image)(image[i:i+patch_size,j:j+patch_size],mask[i:i+patch_size,j:j+patch_size]))
    with ProgressBar():
        res_returned=dask.compute(*res,scheduler="processes")
    img_new=np.ones(image.shape).astype(np.uint8)*255
    for k in tqdm.trange(len(coords)):
        i,j=coords[k]
        img_new[i:i+patch_size,j:j+patch_size]=res_returned[k]
    return img_new

def stain_norm_pipeline(svs,npy_out,compression=10,patch_size=1024):
    np.save(npy_out,stain_norm(svs,compression,patch_size))

if __name__=="__main__":
    fire.Fire(stain_norm_pipeline)
