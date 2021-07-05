"""
utils.py
=======================
General utilities that still need to be broken up into preprocessing, machine learning input preparation, and output submodules.
"""

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
from shapely.ops import unary_union, polygonize
from shapely.geometry import MultiPolygon, LineString
import numpy as np
import dask.array as da
import dask
import openslide
from openslide import deepzoom
#import xarray as xr, sparse
import pickle
import copy

import nonechucks as nc

from nonechucks import SafeDataLoader as DataLoader

def load_sql_df(sql_file, patch_size):
	"""Load pandas dataframe from SQL, accessing particular patch size within SQL.

	Parameters
	----------
	sql_file:str
		SQL db.
	patch_size:int
		Patch size.

	Returns
	-------
	dataframe
		Patch level information.

	"""
	conn = sqlite3.connect(sql_file)
	df=pd.read_sql('select * from "{}";'.format(patch_size),con=conn)
	conn.close()
	return df

def df2sql(df, sql_file, patch_size, mode='replace'):
	"""Write dataframe containing patch level information to SQL db.

	Parameters
	----------
	df:dataframe
		Dataframe containing patch information.
	sql_file:str
		SQL database.
	patch_size:int
		Size of patches.
	mode:str
		Replace or append.

	"""
	conn = sqlite3.connect(sql_file)
	df.set_index('index').to_sql(str(patch_size), con=conn, if_exists=mode)
	conn.close()


#########

# https://github.com/qupath/qupath/wiki/Supported-image-formats
def svs2dask_array(svs_file, tile_size=1000, overlap=0, remove_last=True, allow_unknown_chunksizes=False):
	"""Convert SVS, TIF or TIFF to dask array.

	Parameters
	----------
	svs_file:str
		Image file.
	tile_size:int
		Size of chunk to be read in.
	overlap:int
		Do not modify, overlap between neighboring tiles.
	remove_last:bool
		Remove last tile because it has a custom size.
	allow_unknown_chunksizes: bool
		Allow different chunk sizes, more flexible, but slowdown.

	Returns
	-------
	dask.array
		Dask Array.

	>>> arr=svs2dask_array(svs_file, tile_size=1000, overlap=0, remove_last=True, allow_unknown_chunksizes=False)
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
	"""Convert SVS, TIF, TIFF to NPY.

	Parameters
	----------
	input_dir:str
		Output file dir.
	basename:str
		Basename of output file
	svs_file:str
		SVS, TIF, TIFF file input.

	Returns
	-------
	str
		NPY output file.
	"""
	npy_out_file = join(input_dir,'{}.npy'.format(basename))
	arr = svs2dask_array(svs_file)
	np.save(npy_out_file,arr.compute())
	return npy_out_file

def load_image(svs_file):
	"""Load SVS, TIF, TIFF

	Parameters
	----------
	svs_file:type
		Description of parameter `svs_file`.

	Returns
	-------
	type
		Description of returned object.
	"""
	im = Image.open(svs_file)
	return np.transpose(np.array(im),(1,0)), im.size

def create_purple_mask(arr, img_size=None, sparse=True):
	"""Create a gray scale intensity mask. This will be changed soon to support other thresholding QC methods.

	Parameters
	----------
	arr:dask.array
		Dask array containing image information.
	img_size:int
		Deprecated.
	sparse:bool
		Deprecated

	Returns
	-------
	dask.array
		Intensity, grayscale array over image.

	"""
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
	"""Optional add intensity mask to the dask array.

	Parameters
	----------
	arr:dask.array
		Image data.

	Returns
	-------
	array
		Image data with intensity added as forth channel.

	"""
	return np.concatenate((arr,create_purple_mask(arr)),axis=0)

def create_sparse_annotation_arrays(xml_file, img_size, annotations=[]):
	"""Convert annotation xml to shapely objects and store in dictionary.

	Parameters
	----------
	xml_file:str
		XML file containing annotations.
	img_size:int
		Deprecated.
	annotations:list
		Annotations to look for in xml export.

	Returns
	-------
	dict
		Dictionary with annotation-shapely object pairs.

	"""
	interior_points_dict = {annotation:parse_coord_return_boxes(xml_file, annotation_name = annotation, return_coords = False) for annotation in annotations}#grab_interior_points(xml_file, img_size, annotations=annotations) if annotations else {}
	return {annotation:interior_points_dict[annotation] for annotation in annotations}#sparse.COO.from_scipy_sparse((sps.coo_matrix(interior_points_dict[annotation],img_size, dtype=np.uint8) if interior_points_dict[annotation] not None else sps.coo_matrix(img_size, dtype=np.uint8)).tocsr()) for annotation in annotations} # [sps.coo_matrix(img_size, dtype=np.uint8)]+

def load_process_image(svs_file, xml_file=None, npy_mask=None, annotations=[]):
	"""Load SVS-like image (including NPY), segmentation/classification annotations, generate dask array and dictionary of annotations.

	Parameters
	----------
	svs_file:str
		Image file
	xml_file:str
		Annotation file.
	npy_mask:array
		Numpy segmentation mask.
	annotations:list
		List of annotations in xml.

	Returns
	-------
	array
		Dask array of image.
	dict
		Annotation masks.

	"""
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
	"""Saves dask array image, dictionary of annotations to zarr and pickle respectively.

	Parameters
	----------
	arr:array
		Image.
	masks:dict
		Dictionary of annotation shapes.
	out_zarr:str
		Zarr output file for image.
	out_pkl:str
		Pickle output file.
	"""
	arr.astype('uint8').to_zarr(out_zarr, overwrite=True)
	pickle.dump(masks,open(out_pkl,'wb'))

	#dataset.to_netcdf(out_netcdf, compute=False)
	#pickle.dump(dataset, open(out_pkl,'wb'), protocol=-1)

def run_preprocessing_pipeline(svs_file, xml_file=None, npy_mask=None, annotations=[], out_zarr='output_zarr.zarr', out_pkl='output.pkl'):
	"""Run preprocessing pipeline. Store image into zarr format, segmentations maintain as npy, and xml annotations as pickle.

	Parameters
	----------
	svs_file:str
		Input image file.
	xml_file:str
		Input annotation file.
	npy_mask:str
		NPY segmentation mask.
	annotations:list
		List of annotations.
	out_zarr:str
		Output zarr for image.
	out_pkl:str
		Output pickle for annotations.
	"""
	#save_dataset(load_process_image(svs_file, xml_file, npy_mask, annotations), out_netcdf)
	arr, masks = load_process_image(svs_file, xml_file, npy_mask, annotations)
	save_dataset(arr, masks,out_zarr, out_pkl)

###################

def adjust_mask(mask_file, dask_img_array_file, out_npy, n_neighbors):
	"""Fixes segmentation masks to reduce coarse annotations over empty regions.

	Parameters
	----------
	mask_file:str
		NPY segmentation mask.
	dask_img_array_file:str
		Dask image file.
	out_npy:str
		Output numpy file.
	n_neighbors:int
		Number nearest neighbors for dilation and erosion of mask from background to not background.

	Returns
	-------
	str
		Output numpy file.

	"""
	from dask_image.ndmorph import binary_opening
	from dask.distributed import Client
	#c=Client()
	dask_img_array=da.from_zarr(dask_img_array_file)
	mask=npy2da(mask_file)
	is_tissue_mask = mask>0.
	is_tissue_mask_img=((dask_img_array[...,0]>200.) & (dask_img_array[...,1]>200.)& (dask_img_array[...,2]>200.)) == 0
	opening=binary_opening(is_tissue_mask_img,structure=da.ones((n_neighbors,n_neighbors)))#,mask=is_tissue_mask)
	mask[(opening==0)&(is_tissue_mask==1)]=0
	np.save(out_npy,mask.compute())
	#c.close()
	return out_npy

###################

def process_svs(svs_file, xml_file, annotations=[], output_dir='./'):
	"""Store images into npy format and store annotations into pickle dictionary.

	Parameters
	----------
	svs_file:str
		Image file.
	xml_file:str
		Annotations file.
	annotations:list
		List of annotations in image.
	output_dir:str
		Output directory.
	"""
	os.makedirs(output_dir,exist_ok=True)
	basename = svs_file.split('/')[-1].split('.')[0]
	arr, masks = load_process_image(svs_file, xml_file)
	np.save(join(output_dir,'{}.npy'.format(basename)),arr)
	pickle.dump(masks, open(join(output_dir,'{}.pkl'.format(basename)),'wb'), protocol=-1)

####################

def load_dataset(in_zarr, in_pkl):
	"""Load ZARR image and annotations pickle.

	Parameters
	----------
	in_zarr:str
		Input image.
	in_pkl:str
		Input annotations.

	Returns
	-------
	dask.array
		Image array.
	dict
		Annotations dictionary.

	"""
	return da.from_zarr(in_zarr), pickle.load(open(in_pkl,'rb'))#xr.open_dataset(in_netcdf)

def is_valid_patch(xs,ys,patch_size,purple_mask,intensity_threshold,threshold=0.5):
	"""Deprecated, computes whether patch is valid."""
	print(xs,ys)
	return (purple_mask[xs:xs+patch_size,ys:ys+patch_size]>=intensity_threshold).mean() > threshold

def fix_polygon(poly):
	if not poly.is_valid:
		#print(poly.exterior.coords.xy)

		poly=LineString(np.vstack(poly.exterior.coords.xy).T)
		poly=unary_union(LineString(poly.coords[:] + poly.coords[0:1]))
		#arr.geometry = arr.buffer(0)
		poly = [p for p in polygonize(poly)]
	else:
		poly = [poly]
	return poly

#@pysnooper.snoop("extract_patch.log")
def extract_patch_information(basename, input_dir='./', annotations=[], threshold=0.5, patch_size=224, generate_finetune_segmentation=False, target_class=0, intensity_threshold=100., target_threshold=0., adj_mask='', basic_preprocess=False, tries=0):
	"""Final step of preprocessing pipeline. Break up image into patches, include if not background and of a certain intensity, find area of each annotation type in patch, spatial information, image ID and dump data to SQL table.

	Parameters
	----------
	basename:str
		Patient ID.
	input_dir:str
		Input directory.
	annotations:list
		List of annotations to record, these can be different tissue types, must correspond with XML labels.
	threshold:float
		Value between 0 and 1 that indicates the minimum amount of patch that musn't be background for inclusion.
	patch_size:int
		Patch size of patches; this will become one of the tables.
	generate_finetune_segmentation:bool
		Deprecated.
	target_class:int
		Number of segmentation classes desired, from 0th class to target_class-1 will be annotated in SQL.
	intensity_threshold:float
		Value between 0 and 255 that represents minimum intensity to not include as background. Will be modified with new transforms.
	target_threshold:float
		Deprecated.
	adj_mask:str
		Adjusted mask if performed binary opening operations in previous preprocessing step.
	basic_preprocess:bool
		Do not store patch level information.
	tries:int
		Number of tries in case there is a Dask timeout, run again.

	Returns
	-------
	dataframe
		Patch information.

	"""
	#from collections import OrderedDict
	#annotations=OrderedDict(annotations)
	#from dask.multiprocessing import get
	import dask
	import time
	from dask import dataframe as dd
	import dask.array as da
	import multiprocessing
	from shapely.ops import unary_union
	from shapely.geometry import MultiPolygon
	from itertools import product
	from functools import reduce
	#from distributed import Client,LocalCluster
	max_tries=4
	kargs=dict(basename=basename, input_dir=input_dir, annotations=annotations, threshold=threshold, patch_size=patch_size, generate_finetune_segmentation=generate_finetune_segmentation, target_class=target_class, intensity_threshold=intensity_threshold, target_threshold=target_threshold, adj_mask=adj_mask, basic_preprocess=basic_preprocess, tries=tries)
	try:
		#,
		#						'distributed.scheduler.allowed-failures':20,
		#						'num-workers':20}):
		#cluster=LocalCluster()
		#cluster.adapt(minimum=10, maximum=100)
		#cluster = LocalCluster(threads_per_worker=1, n_workers=20, memory_limit="80G")
		#client=Client()#Client(cluster)#processes=True)#cluster,

		arr, masks = load_dataset(join(input_dir,'{}.zarr'.format(basename)),join(input_dir,'{}_mask.pkl'.format(basename)))
		if 'annotations' in masks:
			segmentation = True

			#if generate_finetune_segmentation:
			segmentation_mask = npy2da(join(input_dir,'{}_mask.npy'.format(basename)) if not adj_mask else adj_mask)
		else:
			segmentation = False
			annotations=list(annotations)
			print(annotations)
			#masks=np.load(masks['annotations'])
		#npy_file = join(input_dir,'{}.npy'.format(basename))
		purple_mask = create_purple_mask(arr)
		x_max = float(arr.shape[0])
		y_max = float(arr.shape[1])
		x_steps = int((x_max-patch_size) / patch_size )
		y_steps = int((y_max-patch_size) / patch_size )
		for annotation in annotations:
			if masks[annotation]:
				masks[annotation]=list(reduce(lambda x,y: x+y, [fix_polygon(poly) for poly in masks[annotation]]))
			try:
				masks[annotation]=[unary_union(masks[annotation])] if masks[annotation] else []
			except:
				masks[annotation]=[MultiPolygon(masks[annotation])] if masks[annotation] else []
		patch_info=pd.DataFrame([([basename,i*patch_size,j*patch_size,patch_size,'NA']+[0.]*(target_class if segmentation else len(annotations))) for i,j in product(range(x_steps+1),range(y_steps+1))],columns=(['ID','x','y','patch_size','annotation']+(annotations if not segmentation else list([str(i) for i in range(target_class)]))))#[dask.delayed(return_line_info)(i,j) for (i,j) in product(range(x_steps+1),range(y_steps+1))]
		if basic_preprocess:
			patch_info=patch_info.iloc[:,:4]
		valid_patches=[]
		for xs,ys in patch_info[['x','y']].values.tolist():
			valid_patches.append(((purple_mask[xs:xs+patch_size,ys:ys+patch_size]>=intensity_threshold).mean() > threshold) if intensity_threshold > 0 else True) # dask.delayed(is_valid_patch)(xs,ys,patch_size,purple_mask,intensity_threshold,threshold)
		valid_patches=np.array(da.compute(*valid_patches))
		print('Valid Patches Complete')
		#print(valid_patches)
		patch_info=patch_info.loc[valid_patches]
		if not basic_preprocess:
			area_info=[]
			if segmentation:
				patch_info.loc[:,'annotation']='segment'
				for xs,ys in patch_info[['x','y']].values.tolist():
					xf=xs+patch_size
					yf=ys+patch_size
					#print(xs,ys)
					area_info.append(da.histogram(segmentation_mask[xs:xf,ys:yf],range=[0,target_class-1],bins=target_class)[0])
					#area_info.append(dask.delayed(seg_line)(xs,ys,patch_size,segmentation_mask,target_class))
			else:
				for xs,ys in patch_info[['x','y']].values.tolist():
					area_info.append([dask.delayed(is_coords_in_box)([xs,ys],patch_size,masks[annotation]) for annotation in annotations])
			#area_info=da.concatenate(area_info,axis=0).compute()
			area_info=np.array(dask.compute(*area_info)).astype(float)#da.concatenate(area_info,axis=0).compute(dtype=np.float16,scheduler='threaded')).astype(np.float16)
			print('Area Info Complete')
			area_info = area_info/(patch_size**2)
			patch_info.iloc[:,5:]=area_info
			#print(patch_info.dtypes)
			annot=list(patch_info.iloc[:,5:])
			patch_info.loc[:,'annotation']=np.vectorize(lambda i: annot[patch_info.iloc[i,5:].values.argmax()])(np.arange(patch_info.shape[0]))#patch_info[np.arange(target_class).astype(str).tolist()].values.argmax(1).astype(str)
			#client.close()
	except Exception as e:
		print(e)
		kargs['tries']+=1
		if kargs['tries']==max_tries:
			raise Exception('Exceeded past maximum number of tries.')
		else:
			print('Restarting preprocessing again.')
			extract_patch_information(**kargs)
	print(patch_info)
	return patch_info

def generate_patch_pipeline(basename, input_dir='./', annotations=[], threshold=0.5, patch_size=224, out_db='patch_info.db', generate_finetune_segmentation=False, target_class=0, intensity_threshold=100., target_threshold=0., adj_mask='', basic_preprocess=False):
	"""Find area coverage of each annotation in each patch and store patch information into SQL db.

	Parameters
	----------
	basename:str
		Patient ID.
	input_dir:str
		Input directory.
	annotations:list
		List of annotations to record, these can be different tissue types, must correspond with XML labels.
	threshold:float
		Value between 0 and 1 that indicates the minimum amount of patch that musn't be background for inclusion.
	patch_size:int
		Patch size of patches; this will become one of the tables.
	out_db:str
		Output SQL database.
	generate_finetune_segmentation:bool
		Deprecated.
	target_class:int
		Number of segmentation classes desired, from 0th class to target_class-1 will be annotated in SQL.
	intensity_threshold:float
		Value between 0 and 255 that represents minimum intensity to not include as background. Will be modified with new transforms.
	target_threshold:float
		Deprecated.
	adj_mask:str
		Adjusted mask if performed binary opening operations in previous preprocessing step.
	basic_preprocess:bool
		Do not store patch level information.
	"""
	patch_info = extract_patch_information(basename, input_dir, annotations, threshold, patch_size, generate_finetune_segmentation=generate_finetune_segmentation, target_class=target_class, intensity_threshold=intensity_threshold, target_threshold=target_threshold, adj_mask=adj_mask, basic_preprocess=basic_preprocess)
	conn = sqlite3.connect(out_db)
	patch_info.to_sql(str(patch_size), con=conn, if_exists='append')
	conn.close()


# now output csv
def save_all_patch_info(basenames, input_dir='./', annotations=[], threshold=0.5, patch_size=224, output_pkl='patch_info.pkl'):
	"""Deprecated."""
	df=pd.concat([extract_patch_information(basename, input_dir, annotations, threshold, patch_size) for basename in basenames]).reset_index(drop=True)
	df.to_pickle(output_pkl)

#########


def create_train_val_test(train_val_test_pkl, input_info_db, patch_size):
	"""Create dataframe that splits slides into training validation and test.

	Parameters
	----------
	train_val_test_pkl:str
		Pickle for training validation and test slides.
	input_info_db:str
		Patch information SQL database.
	patch_size:int
		Patch size looking to access.

	Returns
	-------
	dataframe
		Train test validation splits.

	"""
	if os.path.exists(train_val_test_pkl):
		IDs = pd.read_pickle(train_val_test_pkl)
	else:
		conn = sqlite3.connect(input_info_db)
		df=pd.read_sql('select * from "{}";'.format(patch_size),con=conn)
		conn.close()
		IDs=df['ID'].unique()
		IDs=pd.DataFrame(IDs,columns=['ID'])
		IDs_train, IDs_test = train_test_split(IDs)
		IDs_train, IDs_val = train_test_split(IDs_train)
		IDs_train['set']='train'
		IDs_val['set']='val'
		IDs_test['set']='test'
		IDs=pd.concat([IDs_train,IDs_val,IDs_test])
		IDs.to_pickle(train_val_test_pkl)
	return IDs

def modify_patch_info(input_info_db='patch_info.db', slide_labels=pd.DataFrame(), pos_annotation_class='', patch_size=224, segmentation=False, other_annotations=[], target_segmentation_class=-1, target_threshold=0., classify_annotations=False):
	"""Modify the patch information to get ready for deep learning, incorporate whole slide labels if needed.

	Parameters
	----------
	input_info_db:str
		SQL DB file.
	slide_labels:dataframe
		Dataframe with whole slide labels.
	pos_annotation_class:str
		Tissue/annotation label to label with whole slide image label, if not supplied, any slide's patches receive the whole slide label.
	patch_size:int
		Patch size.
	segmentation:bool
		Segmentation?
	other_annotations:list
		Other annotations to access from patch information.
	target_segmentation_class:int
		Segmentation class to threshold.
	target_threshold:float
		Include patch if patch has target area greater than this.
	classify_annotations:bool
		Classifying annotations for pretraining, or final model?

	Returns
	-------
	dataframe
		Modified patch information.

	"""
	conn = sqlite3.connect(input_info_db)
	df=pd.read_sql('select * from "{}";'.format(patch_size),con=conn)
	conn.close()
	#print(df)
	df=df.drop_duplicates()
	df=df.loc[np.isin(df['ID'],slide_labels.index)]
	#print(classify_annotations)
	if not segmentation:
		if classify_annotations:
			targets=df['annotation'].unique().tolist()
			if len(targets)==1:
				targets=list(df.iloc[:,5:])
		else:
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
				slide_bool=((df['ID']==slide) & df[pos_annotation_class]>0.) if pos_annotation_class else (df['ID']==slide) # (df['annotation']==pos_annotation_class)
				if slide_bool.sum():
					df.loc[slide_bool,targets] = slide_labels.loc[slide,targets].values#1.
		df['area']=np.vectorize(lambda i: df.iloc[i][df.iloc[i]['annotation']])(np.arange(df.shape[0]))
		if 'area' in list(df) and target_threshold>0.:
			df=df.loc[df['area']>=target_threshold]
	else:
		df['target']=0.
		if target_segmentation_class >=0:
			df=df.loc[df[str(target_segmentation_class)]>=target_threshold]
	return df

def npy2da(npy_file):
	"""Numpy to dask array.

	Parameters
	----------
	npy_file:str
		Input npy file.

	Returns
	-------
	dask.array
		Converted numpy array to dask.

	"""
	return da.from_array(np.load(npy_file, mmap_mode = 'r+'))

def grab_interior_points(xml_file, img_size, annotations=[]):
	"""Deprecated."""
	interior_point_dict = {}
	for annotation in annotations:
		try:
			interior_point_dict[annotation] = parse_coord_return_boxes(xml_file, annotation, return_coords = False) # boxes2interior(img_size,
		except:
			interior_point_dict[annotation] = []#np.array([[],[]])
	return interior_point_dict

def boxes2interior(img_size, polygons):
	"""Deprecated."""
	img = Image.new('L', img_size, 0)
	for polygon in polygons:
		ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
	mask = np.array(img).nonzero()
	#mask = (np.ones(len(mask[0])),mask)
	return mask

def parse_coord_return_boxes(xml_file, annotation_name = '', return_coords = False):
	"""Get list of shapely objects for each annotation in the XML object.

	Parameters
	----------
	xml_file:str
		Annotation file.
	annotation_name:str
		Name of xml annotation.
	return_coords:bool
		Just return list of coords over shapes.

	Returns
	-------
	list
		List of shapely objects.

	"""
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
	"""Get area of annotation in patch.

	Parameters
	----------
	coords:array
		X,Y coordinates of patch.
	patch_size:int
		Patch size.
	boxes:list
		Shapely objects for annotations.

	Returns
	-------
	float
		Area of annotation type.

	"""
	if len(boxes):
		points=Polygon(np.array([[0,0],[1,0],[1,1],[0,1]])*patch_size+coords)
		area=points.intersection(boxes[0]).area#any(list(map(lambda x: x.intersects(points),boxes)))#return_image_coord(nx=nx,ny=ny,xi=xi,yi=yi, output_point=output_point)
	else:
		area=0.
	return area

def is_image_in_boxes(image_coord_dict, boxes):
	"""Find if image intersects with annotations.

	Parameters
	----------
	image_coord_dict:dict
		Dictionary of patches.
	boxes:list
		Shapely annotation shapes.

	Returns
	-------
	dict
		Dictionary of whether image intersects with any of the annotations.

	"""
	return {image: any(list(map(lambda x: x.intersects(image_coord_dict[image]),boxes))) for image in image_coord_dict}

def images2coord_dict(images, output_point=False):
	"""Deprecated"""
	return {image: image2coords(image, output_point) for image in images}

def dir2images(image_dir):
	"""Deprecated"""
	return glob.glob(join(image_dir,'*.jpg'))

def return_image_in_boxes_dict(image_dir, xml_file, annotation=''):
	"""Deprecated"""
	boxes = parse_coord_return_boxes(xml_file, annotation)
	images = dir2images(image_dir)
	coord_dict = images2coord_dict(images)
	return is_image_in_boxes(image_coord_dict=coord_dict,boxes=boxes)

def image2coords(image_file, output_point=False):
	"""Deprecated."""
	nx,ny,yi,xi = np.array(image_file.split('/')[-1].split('.')[0].split('_')[1:]).astype(int).tolist()
	return return_image_coord(nx=nx,ny=ny,xi=xi,yi=yi, output_point=output_point)

def retain_images(image_dir,xml_file, annotation=''):
	"""Deprecated"""
	image_in_boxes_dict=return_image_in_boxes_dict(image_dir,xml_file, annotation)
	return [img for img in image_in_boxes_dict if image_in_boxes_dict[img]]

def return_image_coord(nx=0,ny=0,xl=3333,yl=3333,xi=0,yi=0,xc=3,yc=3,dimx=224,dimy=224, output_point=False):
	"""Deprecated"""
	if output_point:
		return np.array([xc,yc])*np.array([nx*xl+xi+dimx/2,ny*yl+yi+dimy/2])
	else:
		static_point = np.array([nx*xl+xi,ny*yl+yi])
		points = np.array([(np.array([xc,yc])*(static_point+np.array(new_point))).tolist() for new_point in [[0,0],[dimx,0],[dimx,dimy],[0,dimy]]])
		return Polygon(points)#Point(*((np.array([xc,yc])*np.array([nx*xl+xi+dimx/2,ny*yl+yi+dimy/2])).tolist())) # [::-1]

def fix_name(basename):
	"""Fixes illegitimate basename, deprecated."""
	if len(basename) < 3:
		return '{}0{}'.format(*basename)
	return basename

def fix_names(file_dir):
	"""Fixes basenames, deprecated."""
	for filename in glob.glob(join(file_dir,'*')):
		basename = filename.split('/')[-1]
		basename, suffix = basename[:basename.rfind('.')], basename[basename.rfind('.'):]
		if len(basename) < 3:
			new_filename=join(file_dir,'{}0{}{}'.format(*basename,suffix))
			print(filename,new_filename)
			subprocess.call('mv {} {}'.format(filename,new_filename),shell=True)

#######

#@pysnooper.snoop('seg2npy.log')
def segmentation_predictions2npy(y_pred, patch_info, segmentation_map, npy_output, original_patch_size=500, resized_patch_size=256):
	"""Convert segmentation predictions from model to numpy masks.

	Parameters
	----------
	y_pred:list
		List of patch segmentation masks
	patch_info:dataframe
		Patch information from DB.
	segmentation_map:array
		Existing segmentation mask.
	npy_output:str
		Output npy file.
	"""
	import cv2
	import copy
	seg_map_shape=segmentation_map.shape[-2:]
	original_seg_shape=copy.deepcopy(seg_map_shape)
	if resized_patch_size!=original_patch_size:
		seg_map_shape = [int(dim*resized_patch_size/original_patch_size) for dim in seg_map_shape]
	segmentation_map = np.zeros(tuple(seg_map_shape))
	for i in range(patch_info.shape[0]):
		patch_info_i = patch_info.iloc[i]
		ID = patch_info_i['ID']
		xs = patch_info_i['x']
		ys = patch_info_i['y']
		patch_size = patch_info_i['patch_size']
		if resized_patch_size!=original_patch_size:
			xs=int(xs*resized_patch_size/original_patch_size)
			ys=int(ys*resized_patch_size/original_patch_size)
			patch_size=resized_patch_size
		prediction=y_pred[i,...]
		segmentation_map[xs:xs+patch_size,ys:ys+patch_size] = prediction
	if resized_patch_size!=original_patch_size:
		segmentation_map=cv2.resize(segmentation_map.astype(float), dsize=original_seg_shape, interpolation=cv2.INTER_NEAREST)
	os.makedirs(npy_output[:npy_output.rfind('/')],exist_ok=True)
	np.save(npy_output,segmentation_map.astype(np.uint8))
