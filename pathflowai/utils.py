import numpy as np
from bs4 import BeautifulSoup
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import glob
from os.path import join
import plotly.graph_objs as go
import plotly.offline as py
import pandas as pd, numpy as np
import scipy.sparse as sps
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS=1e10
import numpy as np
import scipy.sparse as sps
from os.path import join
import os, subprocess, pandas as pd
import sqlite3
import torch
from torch.utils.data import Dataset#, DataLoader
from sklearn.model_selection import train_test_split
import pysnooper

import numpy as np
import dask.array as da
import dask
import openslide
from openslide import deepzoom
import xarray as xr, sparse
import pickle
import copy

import nonechucks as nc

from nonechucks import SafeDataLoader as DataLoader


def svs2dask_array(svs_file, tile_size=1000, overlap=0, remove_last=True, allow_unknown_chunksizes=False):
	""">>> arr=svs2dask_array(svs_file, tile_size=1000, overlap=0, remove_last=True, allow_unknown_chunksizes=False)
		>>> arr2=arr.compute()
		>>> arr3=to_pil(cv2.resize(arr2, dsize=(1440,700), interpolation=cv2.INTER_CUBIC))
		>>> arr3.save(test_image_name)"""
	img=openslide.open_slide(svs_file)
	gen=deepzoom.DeepZoomGenerator(img, tile_size=tile_size, overlap=overlap, limit_bounds=True)
	max_level = len(gen.level_dimensions)-1
	n_tiles_x, n_tiles_y = gen.level_tiles[max_level]
	get_tile = lambda i,j: np.array(gen.get_tile(max_level,(i,j))).transpose((1,0,2))
	sample_tile = get_tile(0,0)
	sample_tile_shape = sample_tile.shape
	dask_get_tile = dask.delayed(get_tile, pure=True)
	arr=da.concatenate([da.concatenate([da.from_delayed(dask_get_tile(i,j),sample_tile_shape,np.uint8) for j in range(n_tiles_y - (0 if not remove_last else 1))],allow_unknown_chunksizes=allow_unknown_chunksizes,axis=1) for i in range(n_tiles_x - (0 if not remove_last else 1))],allow_unknown_chunksizes=allow_unknown_chunksizes)#.transpose([1,0,2])
	return arr

def img2npy_(input_dir,basename, svs_file):
	npy_out_file = join(input_dir,'{}.npy'.format(basename))
	arr = svs2dask_array(svs_file)
	np.save(npy_out_file,arr.compute())
	return npy_out_file

def load_image(svs_file):
	im = Image.open(svs_file)
	return np.transpose(np.array(im),(1,0)), im.size

def create_purple_mask(arr, img_size=None, sparse=True):#, threshold=100.):
	r,b,g=arr[:,:,0],arr[:,:,1],arr[:,:,2]
	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	#rb_avg = (r+b)/2
	mask= ((255.-gray))# >= threshold)#(r > g - 10) & (b > g - 10) & (rb_avg > g + 20)#np.vectorize(is_purple)(arr).astype(int)
	if 0 and sparse:
		mask = mask.nonzero()
		mask = np.array([mask[0].compute(), mask[1].compute()]).T
		#mask = (np.ones(len(mask[0])),mask)
		#mask = sparse.COO.from_scipy_sparse(sps.coo_matrix(mask, img_size, dtype=np.uint8).tocsr())
	return mask

def add_purple_mask(arr):
	return np.concatenate((arr,create_purple_mask(arr)),axis=0)

def create_sparse_annotation_arrays(xml_file, img_size, annotations=[]):
	interior_points_dict = {annotation:parse_coord_return_boxes(xml_file, annotation_name = annotation, return_coords = False) for annotation in annotations}#grab_interior_points(xml_file, img_size, annotations=annotations) if annotations else {}
	return {annotation:interior_points_dict[annotation] for annotation in annotations}#sparse.COO.from_scipy_sparse((sps.coo_matrix(interior_points_dict[annotation],img_size, dtype=np.uint8) if interior_points_dict[annotation] not None else sps.coo_matrix(img_size, dtype=np.uint8)).tocsr()) for annotation in annotations} # [sps.coo_matrix(img_size, dtype=np.uint8)]+

def load_process_image(svs_file, xml_file=None, npy_mask=None, annotations=[]):
	arr = npy2da(svs_file) if svs_file.endswith('.npy') else svs2dask_array(svs_file, tile_size=1000, overlap=0)#load_image(svs_file)
	img_size = arr.shape[:2]
	masks = {}#{'purple': create_purple_mask(arr,img_size,sparse=False)}
	if xml_file is not None:
		masks.update(create_sparse_annotation_arrays(xml_file, img_size, annotations=annotations))
	if npy_mask is not None:
		masks.update({'annotations':npy_mask})
	#data = dict(image=(['x','y','rgb'],arr),**masks)
	#data_arr = {'image':xr.Variable(['x','y','color'], arr)}
	#purple_arr = {'mask':xr.Variable(['x','y'], masks['purple'])}
	#mask_arr =  {m:xr.Variable(['row','col'],masks[m]) for m in masks if m != 'purple'} if 'annotations' not in annotations else {'annotations':xr.Variable(['x','y'],masks['annotations'])}
	#masks['purple'] = masks['purple'].reshape(*masks['purple'].shape,1)
	#arr = da.concatenate([arr,masks.pop('purple')],axis=2)
	return arr, masks#xr.Dataset.from_dict({k:v for k,v in list(data_arr.items())+list(purple_arr.items())+list(mask_arr.items())})#list(dict(image=data_arr,purple=purple_arr,annotations=mask_arr).items()))#arr, masks

def save_dataset(arr, masks, out_zarr, out_pkl):
	arr.astype('uint8').to_zarr(out_zarr, overwrite=True)
	pickle.dump(masks,open(out_pkl,'wb'))

	#dataset.to_netcdf(out_netcdf, compute=False)
	#pickle.dump(dataset, open(out_pkl,'wb'), protocol=-1)

def run_preprocessing_pipeline(svs_file, xml_file=None, npy_mask=None, annotations=[], out_zarr='output_zarr.zarr', out_pkl='output.pkl'):
	#save_dataset(load_process_image(svs_file, xml_file, npy_mask, annotations), out_netcdf)
	arr, masks = load_process_image(svs_file, xml_file, npy_mask, annotations)
	save_dataset(arr, masks,out_zarr, out_pkl)


###################

def process_svs(svs_file, xml_file, annotations=[], output_dir='./'):
	os.makedirs(output_dir,exist_ok=True)
	basename = svs_file.split('/')[-1].split('.')[0]
	arr, masks = load_process_image(svs_file, xml_file)
	np.save(join(output_dir,'{}.npy'.format(basename)),arr)
	pickle.dump(masks, open(join(output_dir,'{}.pkl'.format(basename)),'wb'), protocol=-1)

####################

def load_dataset(in_zarr, in_pkl):
	return da.from_zarr(in_zarr), pickle.load(open(in_pkl,'rb'))#xr.open_dataset(in_netcdf)

def is_valid_patch(patch_mask,threshold=0.5):
	return patch_mask.mean() > threshold

#@pysnooper.snoop("extract_patch.log")
def extract_patch_information(basename, input_dir='./', annotations=[], threshold=0.5, patch_size=224, generate_finetune_segmentation=False, target_class=0, intensity_threshold=100.):
	#from collections import OrderedDict
	#annotations=OrderedDict(annotations)
	patch_info = []
	arr, masks = load_dataset(join(input_dir,'{}.zarr'.format(basename)),join(input_dir,'{}_mask.pkl'.format(basename)))
	if 'annotations' in masks:
		segmentation = True
		if generate_finetune_segmentation:
			segmentation_mask = npy2da(join(input_dir,'{}.npy'.format(basename)))
	else:
		segmentation = False
		#masks=np.load(masks['annotations'])
	#npy_file = join(input_dir,'{}.npy'.format(basename))
	purple_mask = create_purple_mask(arr)
	x_max = float(arr.shape[0])
	y_max = float(arr.shape[1])
	x_steps = int((x_max-patch_size) / patch_size )
	y_steps = int((y_max-patch_size) / patch_size )
	for i in range(x_steps+1):
		for j in range(y_steps+1):
			xs = i*patch_size
			ys = j*patch_size
			xf = xs + patch_size
			yf = ys + patch_size
			if is_valid_patch((purple_mask[xs:xf,ys:yf]>=intensity_threshold).compute(), threshold):#.compute()
				print(xs,ys, 'valid_patch')
				if segmentation:
					if generate_finetune_segmentation:
						if is_valid_patch((segmentation_mask[xs:xf,ys:yf]==target_class).compute(), 0.):
							patch_info.append([basename,xs,ys,patch_size,'{}'.format(target_class)])
					else:
						patch_info.append([basename,xs,ys,patch_size,'segment'])
				else:
					for annotation in annotations:
						#mask_patch = masks[xs:xf,ys:yf]
						if is_coords_in_box(coords=np.array([xs,ys]),patch_size=patch_size,boxes=masks[annotation]):#is_valid_patch(masks[annotation][xs:xf,ys:yf], threshold):
							patch_info.append([basename,xs,ys,patch_size,annotation])
							break
	patch_info = pd.DataFrame(patch_info,columns=['ID','x','y','patch_size','annotation'])
	return patch_info

def generate_patch_pipeline(basename, input_dir='./', annotations=[], threshold=0.5, patch_size=224, out_db='patch_info.db', generate_finetune_segmentation=False, target_class=0, intensity_threshold=100.):
	patch_info = extract_patch_information(basename, input_dir, annotations, threshold, patch_size, generate_finetune_segmentation=generate_finetune_segmentation, target_class=target_class, intensity_threshold=intensity_threshold)
	conn = sqlite3.connect(out_db)
	patch_info.to_sql(str(patch_size), con=conn, if_exists='append')
	conn.close()


# now output csv
def save_all_patch_info(basenames, input_dir='./', annotations=[], threshold=0.5, patch_size=224, output_pkl='patch_info.pkl'):
	df=pd.concat([extract_patch_information(basename, input_dir, annotations, threshold, patch_size) for basename in basenames]).reset_index(drop=True)
	df.to_pickle(output_pkl)

#########


def create_train_val_test(train_val_test_pkl, input_info_db, patch_size):
	if os.path.exists(train_val_test_pkl):
		IDs = pd.read_pickle(train_val_test_pkl)
	else:
		conn = sqlite3.connect(input_info_db)
		df=pd.read_sql('select * from "{}";'.format(patch_size),con=conn)
		conn.close()
		IDs=df['ID'].unique()
		IDs=pd.DataFrame(IDs,columns=['ID'])
		IDs_train, IDs_test = train_test_split(IDs)
		IDs_train, IDs_val = train_test_split(IDs)
		IDs_train['set']='train'
		IDs_val['set']='val'
		IDs_test['set']='test'
		IDs=pd.concat([IDs_train,IDs_val,IDs_test])
		IDs.to_pickle(train_val_test_pkl)
	return IDs

def modify_patch_info(input_info_db='patch_info.db', slide_labels=pd.DataFrame(), pos_annotation_class='melanocyte', patch_size=224, segmentation=False, other_annotations=[]):
	conn = sqlite3.connect(input_info_db)
	df=pd.read_sql('select * from "{}";'.format(patch_size),con=conn)
	conn.close()
	#print(df)

	df=df.loc[np.isin(df['ID'],slide_labels.index)]
	if not segmentation:
		targets = list(slide_labels)
		if type(pos_annotation_class)==type(''):
			included_annotations = [pos_annotation_class]
		else:
			included_annotations = copy.deepcopy(pos_annotation_class)
		included_annotations.extend(other_annotations)
		df=df[np.isin(df['annotation'],included_annotations)]
		for target in targets:
			df[target]=0.
		for slide in slide_labels.index:
			slide_bool=((df['ID']==slide) & (df['annotation']==pos_annotation_class))
			if slide_bool.sum():
				df.loc[slide_bool,targets] = slide_labels.loc[slide,targets].values#1.
	else:
		df['target']=0.
	return df

def npy2da(npy_file):
	return da.from_array(np.load(npy_file, mmap_mode = 'r+'))

def grab_interior_points(xml_file, img_size, annotations=[]):
    interior_point_dict = {}
    for annotation in annotations:
        try:
            interior_point_dict[annotation] = parse_coord_return_boxes(xml_file, annotation, return_coords = False) # boxes2interior(img_size,
        except:
            interior_point_dict[annotation] = []#np.array([[],[]])
    return interior_point_dict

def boxes2interior(img_size, polygons):
    img = Image.new('L', img_size, 0)
    for polygon in polygons:
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    mask = np.array(img).nonzero()
    #mask = (np.ones(len(mask[0])),mask)
    return mask

def parse_coord_return_boxes(xml_file, annotation_name = 'melanocyte', return_coords = False):
    boxes = []
    xml_data = BeautifulSoup(open(xml_file),'html')
    #print(xml_data.findAll('annotation'))
    #print(xml_data.findAll('Annotation'))
    for annotation in xml_data.findAll('annotation'):
        if annotation['partofgroup'] == annotation_name:
            for coordinates in annotation.findAll('coordinates'):
				# FIXME may need to change x and y coordinates
                coords = [(coordinate['x'],coordinate['y']) for coordinate in coordinates.findAll('coordinate')]
                if return_coords:
                    boxes.append(coords)
                else:
                    boxes.append(Polygon(np.array(coords).astype(np.float)))
    return boxes

def is_coords_in_box(coords,patch_size,boxes):
    points=Polygon(np.array([[0,0],[1,0],[1,1],[0,1]])*patch_size+coords)
    return any(list(map(lambda x: x.intersects(points),boxes)))#return_image_coord(nx=nx,ny=ny,xi=xi,yi=yi, output_point=output_point)

def is_image_in_boxes(image_coord_dict, boxes):
    return {image: any(list(map(lambda x: x.intersects(image_coord_dict[image]),boxes))) for image in image_coord_dict}

def images2coord_dict(images, output_point=False):
    return {image: image2coords(image, output_point) for image in images}

def dir2images(image_dir):
    return glob.glob(join(image_dir,'*.jpg'))

def return_image_in_boxes_dict(image_dir, xml_file, annotation='melanocyte'):
    boxes = parse_coord_return_boxes(xml_file, annotation)
    images = dir2images(image_dir)
    coord_dict = images2coord_dict(images)
    return is_image_in_boxes(image_coord_dict=coord_dict,boxes=boxes)

def image2coords(image_file, output_point=False):
    nx,ny,yi,xi = np.array(image_file.split('/')[-1].split('.')[0].split('_')[1:]).astype(int).tolist()
    return return_image_coord(nx=nx,ny=ny,xi=xi,yi=yi, output_point=output_point)

def retain_images(image_dir,xml_file, annotation='melanocyte'):
    image_in_boxes_dict=return_image_in_boxes_dict(image_dir,xml_file, annotation)
    return [img for img in image_in_boxes_dict if image_in_boxes_dict[img]]

def return_image_coord(nx=0,ny=0,xl=3333,yl=3333,xi=0,yi=0,xc=3,yc=3,dimx=224,dimy=224, output_point=False):
    if output_point:
        return np.array([xc,yc])*np.array([nx*xl+xi+dimx/2,ny*yl+yi+dimy/2])
    else:
        static_point = np.array([nx*xl+xi,ny*yl+yi])
        points = np.array([(np.array([xc,yc])*(static_point+np.array(new_point))).tolist() for new_point in [[0,0],[dimx,0],[dimx,dimy],[0,dimy]]])
        return Polygon(points)#Point(*((np.array([xc,yc])*np.array([nx*xl+xi+dimx/2,ny*yl+yi+dimy/2])).tolist())) # [::-1]

def fix_name(basename):
	if len(basename) < 3:
		return '{}0{}'.format(*basename)
	return basename

def fix_names(file_dir):
	for filename in glob.glob(join(file_dir,'*')):
		basename = filename.split('/')[-1]
		basename, suffix = basename[:basename.rfind('.')], basename[basename.rfind('.'):]
		if len(basename) < 3:
			new_filename=join(file_dir,'{}0{}{}'.format(*basename,suffix))
			print(filename,new_filename)
			subprocess.call('mv {} {}'.format(filename,new_filename),shell=True)
