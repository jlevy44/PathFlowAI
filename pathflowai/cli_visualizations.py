import click
from visualize import PredictionPlotter, plot_image_
import glob, os
#from utils import *
import dask.array as da


CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def visualize():
	pass

@visualize.command()
@click.option('-i', '--input_dir', default='./inputs/', help='Input directory for patches.', type=click.Path(exists=False), show_default=True)
@click.option('-b', '--basename', default='A01', help='Basename of patches.', type=click.Path(exists=False), show_default=True)
@click.option('-p', '--patch_info_file', default='patch_info.db', help='Datbase containing all patches', type=click.Path(exists=False), show_default=True)
@click.option('-ps', '--patch_size', default=224, help='Patch size.',  show_default=True)
@click.option('-x', '--x', default=0, help='X Coordinate of patch.',  show_default=True)
@click.option('-y', '--y', default=0, help='Y coordinate of patch.',  show_default=True)
@click.option('-o', '--outputfname', default='./output_image.png', help='Output extracted image.', type=click.Path(exists=False), show_default=True)
@click.option('-s', '--segmentation', is_flag=True, help='Plot segmentations.', show_default=True)
@click.option('-sc', '--n_segmentation_classes', default=4, help='Number segmentation classes',  show_default=True)
@click.option('-c', '--custom_segmentation', default='', help='Add custom segmentation map from prediction, in npy',  show_default=True)
def extract_patch(input_dir, basename, patch_info_file, patch_size, x, y, outputfname, segmentation, n_segmentation_classes, custom_segmentation):
	dask_arr_dict = {os.path.basename(f).split('.zarr')[0]:da.from_zarr(f) for f in glob.glob(os.path.join(input_dir,'*.zarr')) if os.path.basename(f).split('.zarr')[0] == basename}
	pred_plotter = PredictionPlotter(dask_arr_dict, patch_info_file, compression_factor=3, alpha=0.5, patch_size=patch_size, no_db=True, segmentation=segmentation,n_segmentation_classes=n_segmentation_classes, input_dir=input_dir)
	if custom_segmentation:
		pred_plotter.add_custom_segmentation(basename,custom_segmentation)
	img = pred_plotter.return_patch(basename, x, y, patch_size)
	pred_plotter.output_image(img,outputfname)

@visualize.command()
@click.option('-i', '--image_file', default='./inputs/a.svs', help='Input image file.', type=click.Path(exists=False), show_default=True)
@click.option('-cf', '--compression_factor', default=3., help='How much compress image.',  show_default=True)
@click.option('-o', '--outputfname', default='./output_image.png', help='Output extracted image.', type=click.Path(exists=False), show_default=True)
def plot_image(image_file, compression_factor, outputfname):
	plot_image_(image_file, compression_factor=compression_factor, test_image_name=outputfname)

@visualize.command()
@click.option('-i', '--input_dir', default='./inputs/', help='Input directory for patches.', type=click.Path(exists=False), show_default=True)
@click.option('-b', '--basename', default='A01', help='Basename of patches.', type=click.Path(exists=False), show_default=True)
@click.option('-p', '--patch_info_file', default='patch_info.db', help='Datbase containing all patches', type=click.Path(exists=False), show_default=True)
@click.option('-ps', '--patch_size', default=224, help='Patch size.',  show_default=True)
@click.option('-o', '--outputfname', default='./output_image.png', help='Output extracted image.', type=click.Path(exists=False), show_default=True)
@click.option('-an', '--annotations', is_flag=True, help='Plot annotations instead of predictions.', show_default=True)
@click.option('-cf', '--compression_factor', default=3., help='How much compress image.',  show_default=True)
@click.option('-al', '--alpha', default=0.8, help='How much to give annotations/predictions versus original image.',  show_default=True)
@click.option('-s', '--segmentation', is_flag=True, help='Plot segmentations.', show_default=True)
@click.option('-sc', '--n_segmentation_classes', default=4, help='Number segmentation classes',  show_default=True)
@click.option('-c', '--custom_segmentation', default='', help='Add custom segmentation map from prediction, npy format.',  show_default=True)
@click.option('-ac', '--annotation_col', default='annotation', help='Column of annotations', type=click.Path(exists=False), show_default=True)
@click.option('-sf', '--scaling_factor', default=1., help='Multiply all prediction scores by this amount.',  show_default=True)
@click.option('-tif', '--tif_file', is_flag=True, help='Write to tiff file.',  show_default=True)
def plot_predictions(input_dir,basename,patch_info_file,patch_size,outputfname,annotations, compression_factor, alpha, segmentation, n_segmentation_classes, custom_segmentation, annotation_col, scaling_factor, tif_file):
	dask_arr_dict = {os.path.basename(f).split('.zarr')[0]:da.from_zarr(f) for f in glob.glob(os.path.join(input_dir,'*.zarr')) if os.path.basename(f).split('.zarr')[0] == basename}
	pred_plotter = PredictionPlotter(dask_arr_dict, patch_info_file, compression_factor=compression_factor, alpha=alpha, patch_size=patch_size, no_db=False, plot_annotation=annotations, segmentation=segmentation, n_segmentation_classes=n_segmentation_classes, input_dir=input_dir, annotation_col=annotation_col, scaling_factor=scaling_factor)
	if custom_segmentation:
		pred_plotter.add_custom_segmentation(basename,custom_segmentation)
	img = pred_plotter.generate_image(basename)
	pred_plotter.output_image(img, outputfname, tif_file)

@visualize.command()
@click.option('-i', '--embeddings_file', default='predictions/embeddings.pkl', help='Embeddings.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--plotly_output_file', default='predictions/embeddings.html', help='Plotly output file.', type=click.Path(exists=False), show_default=True)
@click.option('-a', '--annotations', default=[], multiple=True, help='Multiple annotations to color image.', show_default=True)
@click.option('-rb', '--remove_background_annotation', default='', help='If selected, removes 100\% background patches based on this annotation.', type=click.Path(exists=False), show_default=True)
@click.option('-ma', '--max_background_area', default=0.05, help='Max background area before exclusion.',  show_default=True)
@click.option('-b', '--basename', default='', help='Basename of patches.', type=click.Path(exists=False), show_default=True)
@click.option('-nn', '--n_neighbors', default=8, help='Number nearest neighbors.',  show_default=True)
def plot_embeddings(embeddings_file,plotly_output_file, annotations, remove_background_annotation , max_background_area, basename, n_neighbors):
	import torch
	from umap import UMAP
	from visualize import PlotlyPlot
	import pandas as pd, numpy as np
	embeddings_dict=torch.load(embeddings_file)
	embeddings=embeddings_dict['embeddings']
	patch_info=embeddings_dict['patch_info']
	if remove_background_annotation:
		removal_bool=(patch_info[remove_background_annotation]<=(1.-max_background_area)).values
		patch_info=patch_info.loc[removal_bool]
		embeddings=embeddings.loc[removal_bool]
	if basename:
		removal_bool=(patch_info['ID']==basename).values
		patch_info=patch_info.loc[removal_bool]
		embeddings=embeddings.loc[removal_bool]
	if annotations:
		annotations=np.array(annotations)
		if len(annotations)>1:
			embeddings.loc[:,'ID']=np.vectorize(lambda i: annotations[np.argmax(patch_info.iloc[i][annotations].values)])(np.arange(embeddings.shape[0]))
		else:
			embeddings.loc[:,'ID']=patch_info[annotations].values
	umap=UMAP(n_components=3,n_neighbors=n_neighbors)
	t_data=pd.DataFrame(umap.fit_transform(embeddings.iloc[:,:-1].values),columns=['x','y','z'],index=embeddings.index)
	t_data['color']=embeddings['ID'].values
	t_data['name']=embeddings.index.values
	pp=PlotlyPlot()
	pp.add_plot(t_data)
	pp.plot(plotly_output_file)

@visualize.command()
@click.option('-m', '--model_pkl', default='', help='Plotly output file.', type=click.Path(exists=False), show_default=True)
@click.option('-bs', '--batch_size', default=32, help='Batch size.',  show_default=True)
@click.option('-o', '--outputfilename', default='predictions/shap_plots.png', help='SHAPley visualization.', type=click.Path(exists=False), show_default=True)
@click.option('-mth', '--method', default='deep', help='Method of explaining.', type=click.Choice(['deep','gradient']), show_default=True)
@click.option('-l', '--local_smoothing', default=0.0, help='Local smoothing of SHAP scores.',  show_default=True)
@click.option('-ns', '--n_samples', default=32, help='Number shapley samples for shapley regression (gradient explainer).',  show_default=True)
def shapley_plot(model_pkl, batch_size, outputfilename, method='deep', local_smoothing=0.0, n_samples=20):
	from visualize import plot_shap
	import torch
	from datasets import get_data_transforms
	model_dict=torch.load(model_pkl)
	model_dict['dataset_opts']['transformers']=get_data_transforms(**model_dict['transform_opts'])
	plot_shap(model_dict['model'], model_dict['dataset_opts'], model_dict['transform_opts'], batch_size, outputfilename, method=method, local_smoothing=local_smoothing, n_samples=n_samples)

@visualize.command()
@click.option('-i', '--input_dir', default='./inputs/', help='Input directory for patches.', type=click.Path(exists=False), show_default=True)
@click.option('-e', '--embeddings_file', default='predictions/embeddings.pkl', help='Embeddings.', type=click.Path(exists=False), show_default=True)
@click.option('-b', '--basename', default='A01', help='Basename of patches.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--outputfilename', default='predictions/shap_plots.png', help='Embedding visualization.', type=click.Path(exists=False), show_default=True)
@click.option('-mpl', '--mpl_scatter', is_flag=True, help='Plot segmentations.', show_default=True)
@click.option('-rb', '--remove_background_annotation', default='', help='If selected, removes 100\% background patches based on this annotation.', type=click.Path(exists=False), show_default=True)
@click.option('-ma', '--max_background_area', default=0.05, help='Max background area before exclusion.',  show_default=True)
@click.option('-z', '--zoom', default=0.05, help='Size of images.',  show_default=True)
@click.option('-nn', '--n_neighbors', default=8, help='Number nearest neighbors.',  show_default=True)
def plot_image_umap_embeddings(input_dir,embeddings_file,basename,outputfilename,mpl_scatter, remove_background_annotation, max_background_area, zoom, n_neighbors):
	from visualize import plot_umap_images
	dask_arr_dict = {os.path.basename(f).split('.zarr')[0]:da.from_zarr(f) for f in glob.glob(os.path.join(input_dir,'*.zarr'))  if os.path.basename(f).split('.zarr')[0] == basename}
	plot_umap_images(dask_arr_dict, embeddings_file, ID=basename, cval=1., image_res=300., outputfname=outputfilename, mpl_scatter=mpl_scatter, remove_background_annotation=remove_background_annotation, max_background_area=max_background_area, zoom=zoom, n_neighbors=n_neighbors)

if __name__ == '__main__':
	visualize()
