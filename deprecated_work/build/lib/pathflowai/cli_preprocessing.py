import argparse
import os
from os.path import join
from pathflowai.utils import run_preprocessing_pipeline, generate_patch_pipeline, img2npy_
import click
import dask
import time

CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def preprocessing():
	pass

def output_if_exists(filename):
	"""Returns file name if the file exists

	Parameters
	----------
	filename : str
		File in question.

	Returns
	-------
	str
		Filename.

	"""

	if os.path.exists(filename):
		return filename
	return None

@preprocessing.command()
@click.option('-npy', '--img2npy', is_flag=True, help='Image to numpy for faster read.', show_default=True)
@click.option('-b', '--basename', default='A01', help='Basename of patches.', type=click.Path(exists=False), show_default=True)
@click.option('-i', '--input_dir', default='./inputs/', help='Input directory for patches.', type=click.Path(exists=False), show_default=True)
@click.option('-a', '--annotations', default=[], multiple=True, help='Annotations in image in order.', type=click.Path(exists=False), show_default=True)
@click.option('-pr', '--preprocess', is_flag=True, help='Run preprocessing pipeline.', show_default=True)
@click.option('-pa', '--patches', is_flag=True, help='Add patches to SQL.', show_default=True)
@click.option('-t', '--threshold', default=0.05, help='Threshold to remove non-purple slides.',  show_default=True)
@click.option('-ps', '--patch_size', default=224, help='Patch size.',  show_default=True)
@click.option('-it', '--intensity_threshold', default=100., help='Intensity threshold to rate a pixel as non-white.',  show_default=True)
@click.option('-g', '--generate_finetune_segmentation', is_flag=True, help='Generate patches for one segmentation mask class for targeted finetuning.', show_default=True)
@click.option('-tc', '--target_segmentation_class', default=0, help='Segmentation Class to finetune on, output patches to another db.',  show_default=True)
@click.option('-tt', '--target_threshold', default=0., help='Threshold to include target for segmentation if saving one class.',  show_default=True)
@click.option('-odb', '--out_db', default='./patch_info.db', help='Output patch database.', type=click.Path(exists=False), show_default=True)
@click.option('-am', '--adjust_mask', is_flag=True, help='Remove additional background regions from annotation mask.', show_default=True)
@click.option('-nn', '--n_neighbors', default=5, help='If adjusting mask, number of neighbors connectivity to remove.',  show_default=True)
@click.option('-bp', '--basic_preprocess', is_flag=True, help='Basic preprocessing pipeline, annotation areas are not saved. Used for benchmarking tool against comparable pipelines', show_default=True)
def preprocess_pipeline(img2npy,basename,input_dir,annotations,preprocess,patches,threshold,patch_size, intensity_threshold, generate_finetune_segmentation, target_segmentation_class, target_threshold, out_db, adjust_mask, n_neighbors, basic_preprocess):
	"""Preprocessing pipeline that accomplishes 3 things. 1: storage into ZARR format, 2: optional mask adjustment, 3: storage of patch-level information into SQL DB"""

	for ext in ['.npy','.svs','.tiff','.tif', '.vms', '.vmu', '.ndpi', '.scn', '.mrxs', '.svslide', '.bif', '.jpeg', '.png']:
		svs_file = output_if_exists(join(input_dir,'{}{}'.format(basename,ext)))
		if svs_file != None:
			break

	if img2npy and not svs_file.endswith('.npy'):
		svs_file = img2npy_(input_dir,basename, svs_file)

	xml_file = output_if_exists(join(input_dir,'{}.xml'.format(basename)))
	npy_mask = output_if_exists(join(input_dir,'{}_mask.npy'.format(basename)))
	out_zarr = join(input_dir,'{}.zarr'.format(basename))
	out_pkl = join(input_dir,'{}_mask.pkl'.format(basename))
	adj_npy=''


	start=time.time()
	if preprocess:
		run_preprocessing_pipeline(svs_file=svs_file,
							   xml_file=xml_file,
							   npy_mask=npy_mask,
							   annotations=annotations,
							   out_zarr=out_zarr,
							   out_pkl=out_pkl)
	preprocess_point = time.time()
	print('Data dump took {}'.format(preprocess_point-start))

	if adjust_mask:
		from pathflowai.utils import adjust_mask
		adj_dir=join(input_dir,'adjusted_masks')
		adj_npy=join(adj_dir,os.path.basename(npy_mask))
		os.makedirs(adj_dir,exist_ok=True)
		if not os.path.exists(adj_npy):
			adjust_mask(npy_mask, out_zarr, adj_npy, n_neighbors)
	adjust_point = time.time()
	print('Adjust took {}'.format(adjust_point-preprocess_point))


	if patches: # ADD EXPORT TO SQL, TABLE NAME IS PATCH SIZE
		generate_patch_pipeline(basename,
							input_dir=input_dir,
							annotations=annotations,
							threshold=threshold,
							patch_size=patch_size,
							out_db=out_db,
							generate_finetune_segmentation=generate_finetune_segmentation,
							target_class=target_segmentation_class,
							intensity_threshold=intensity_threshold,
							target_threshold=target_threshold,
							adj_mask=adj_npy,
							basic_preprocess=basic_preprocess)
	patch_point = time.time()
	print('Patches took {}'.format(patch_point-adjust_point))

@preprocessing.command()
@click.option('-i', '--mask_dir', default='./inputs/', help='Input directory for masks.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--output_dir', default='./outputs/', help='Output directory for new masks.', type=click.Path(exists=False), show_default=True)
@click.option('-fr', '--from_annotations', default=[], multiple=True, help='Annotations to switch from.', show_default=True)
@click.option('-to', '--to_annotations', default=[], multiple=True, help='Annotations to switch to.', show_default=True)
def alter_masks(mask_dir, output_dir, from_annotations, to_annotations):
	"""Map list of values to other values in mask."""
	import glob
	from pathflowai.utils import npy2da
	import numpy as np
	from dask.distributed import Client
	assert len(from_annotations)==len(to_annotations)
	c=Client()
	from_annotations=list(map(int,from_annotations))
	to_annotations=list(map(int,to_annotations))
	os.makedirs(output_dir,exist_ok=True)
	masks=glob.glob(join(mask_dir,'*_mask.npy'))
	from_to=list(zip(from_annotations,to_annotations))
	for mask in masks:
		output_mask=join(output_dir,os.path.basename(mask))
		arr=npy2da(mask)
		for fr,to in from_to:
			arr[arr==fr]=to
		np.save(output_mask,arr.compute())

@preprocessing.command()
@click.option('-i', '--input_patch_db', default='patch_info_input.db', help='Input db.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--output_patch_db', default='patch_info_output.db', help='Output db.', type=click.Path(exists=False), show_default=True)
@click.option('-b', '--basename', default='A01', help='Basename.', type=click.Path(exists=False), show_default=True)
@click.option('-ps', '--patch_size', default=224, help='Patch size.',  show_default=True)
def remove_basename_from_db(input_patch_db, output_patch_db, basename, patch_size):
	"""Removes basename/ID from SQL DB."""
	import sqlite3
	import numpy as np, pandas as pd
	os.makedirs(output_patch_db[:output_patch_db.rfind('/')],exist_ok=True)
	conn = sqlite3.connect(input_patch_db)
	df=pd.read_sql('select * from "{}";'.format(patch_size),con=conn)
	conn.close()
	df=df.loc[df['ID']!=basename]
	conn = sqlite3.connect(output_patch_db)
	df.set_index('index').to_sql(str(patch_size), con=conn, if_exists='replace')
	conn.close()


@preprocessing.command()
@click.option('-i', '--input_patch_db', default='patch_info_input.db', help='Input db.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--output_patch_db', default='patch_info_output.db', help='Output db.', type=click.Path(exists=False), show_default=True)
@click.option('-fr', '--from_annotations', default=[], multiple=True, help='Annotations to switch from.', show_default=True)
@click.option('-to', '--to_annotations', default=[], multiple=True, help='Annotations to switch to.', show_default=True)
@click.option('-ps', '--patch_size', default=224, help='Patch size.',  show_default=True)
@click.option('-rb', '--remove_background_annotation', default='', help='If selected, removes 100\% background patches based on this annotation.', type=click.Path(exists=False), show_default=True)
@click.option('-ma', '--max_background_area', default=0.05, help='Max background area before exclusion.',  show_default=True)
def collapse_annotations(input_patch_db, output_patch_db, from_annotations, to_annotations, patch_size, remove_background_annotation, max_background_area):
	"""Adds annotation classes areas to other annotation classes in SQL DB when getting rid of some annotation classes."""
	import sqlite3
	import numpy as np, pandas as pd
	assert len(from_annotations)==len(to_annotations)
	from_annotations=list(map(str,from_annotations))
	to_annotations=list(map(str,to_annotations))
	os.makedirs(output_patch_db[:output_patch_db.rfind('/')],exist_ok=True)
	conn = sqlite3.connect(input_patch_db)
	df=pd.read_sql('select * from "{}";'.format(patch_size),con=conn)
	conn.close()
	from_to=zip(from_annotations,to_annotations)
	if remove_background_annotation:
		df=df.loc[df[remove_background_annotation]<=(1.-max_background_area)]
	for fr,to in from_to:
		df.loc[:,to]+=df[fr]
	df=df[[col for col in list(df) if col not in from_annotations]]
	annotations = list(df.iloc[:,6:])
	df=df.rename(columns={annot:str(i) for i, annot in enumerate(annotations)})
	annotations = list(df.iloc[:,6:])
	df.loc[:,'annotation']=np.vectorize(lambda i: annotations[df.iloc[i,6:].values.argmax()])(np.arange(df.shape[0]))
	df.loc[:,'index']=np.arange(df.shape[0])
	conn = sqlite3.connect(output_patch_db)
	#print(df)
	df.set_index('index').to_sql(str(patch_size), con=conn, if_exists='replace')
	conn.close()


if __name__ == '__main__':
	from dask.distributed import Client
	dask.config.set({'temporary_dir':'tmp/',
					'distributed.worker.local_dir':'tmp/',
					'distributed.scheduler.allowed-failures':20})#'distributed.worker.num-workers':20}):
	c=Client(processes=False)
	preprocessing()
	c.close()
