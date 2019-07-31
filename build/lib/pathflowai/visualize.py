"""
visualize.py
=======================
Plots SHAP outputs, UMAP embeddings, and overlays predictions on top of WSI.
"""

import plotly.graph_objs as go
import plotly.offline as py
import pandas as pd, numpy as np
import networkx as nx
import dask.array as da
from PIL import Image
import matplotlib, matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import seaborn as sns
from os.path import join
sns.set()

class PlotlyPlot:
	"""Creates plotly html plots."""
	def __init__(self):
		self.plots=[]

	def add_plot(self, t_data_df, G=None, color_col='color', name_col='name', xyz_cols=['x','y','z'], size=2, opacity=1.0, custom_colors=[]):
		"""Adds plotting data to be plotted.

		Parameters
		----------
		t_data_df:dataframe
			3-D transformed dataframe.
		G:nx.Graph
			Networkx graph.
		color_col:str
			Column to use to color points.
		name_col:str
			Column to use to name points.
		xyz_cols:list
			3 columns that denote x,y,z coords.
		size:int
			Marker size.
		opacity:float
			Marker opacity.
		custom_colors:list
			Custom colors to supply.
		"""
		plots = []
		x,y,z=tuple(xyz_cols)
		if t_data_df[color_col].dtype == np.float64:
			plots.append(
				go.Scatter3d(x=t_data_df[x], y=t_data_df[y],
							 z=t_data_df[z],
							 name='', mode='markers',
							 marker=dict(color=t_data_df[color_col], size=size, opacity=opacity, colorscale='Viridis',
							 colorbar=dict(title='Colorbar')), text=t_data_df[color_col] if name_col not in list(t_data_df) else t_data_df[name_col]))
		else:
			colors = t_data_df[color_col].unique()
			c = sns.color_palette('hls', len(colors))
			c = np.array(['rgb({})'.format(','.join(((np.array(c_i)*255).astype(int).astype(str).tolist()))) for c_i in c])#c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, len(colors) + 2)]
			if custom_colors:
				c = custom_colors
			color_dict = {name: c[i] for i,name in enumerate(sorted(colors))}

			for name,col in color_dict.items():
				plots.append(
					go.Scatter3d(x=t_data_df[x][t_data_df[color_col]==name], y=t_data_df[y][t_data_df[color_col]==name],
								 z=t_data_df[z][t_data_df[color_col]==name],
								 name=str(name), mode='markers',
								 marker=dict(color=col, size=size, opacity=opacity), text=t_data_df.index[t_data_df[color_col]==name] if 'name' not in list(t_data_df) else t_data_df[name_col][t_data_df[color_col]==name]))
		if G is not None:
			#pos = nx.spring_layout(G,dim=3,iterations=0,pos={i: tuple(t_data.loc[i,['x','y','z']]) for i in range(len(t_data))})
			Xed, Yed, Zed = [], [], []
			for edge in G.edges():
				if edge[0] in t_data_df.index.values and edge[1] in t_data_df.index.values:
					Xed += [t_data_df.loc[edge[0],x], t_data_df.loc[edge[1],x], None]
					Yed += [t_data_df.loc[edge[0],y], t_data_df.loc[edge[1],y], None]
					Zed += [t_data_df.loc[edge[0],z], t_data_df.loc[edge[1],z], None]
			plots.append(go.Scatter3d(x=Xed,
					  y=Yed,
					  z=Zed,
					  mode='lines',
					  line=go.scatter3d.Line(color='rgb(210,210,210)', width=2),
					  hoverinfo='none'
					  ))
		self.plots.extend(plots)

	def plot(self, output_fname, axes_off=False):
		"""Plot embedding of patches to html file.

		Parameters
		----------
		output_fname:str
			Output html file.
		axes_off:bool
			Remove axes.

		"""
		if axes_off:
			fig = go.Figure(data=self.plots,layout=go.Layout(scene=dict(xaxis=dict(title='',autorange=True,showgrid=False,zeroline=False,showline=False,ticks='',showticklabels=False),
				yaxis=dict(title='',autorange=True,showgrid=False,zeroline=False,showline=False,ticks='',showticklabels=False),
				zaxis=dict(title='',autorange=True,showgrid=False,zeroline=False,showline=False,ticks='',showticklabels=False))))
		else:
			fig = go.Figure(data=self.plots)
		py.plot(fig, filename=output_fname, auto_open=False)

def to_pil(arr):
	"""Numpy array to pil.

	Parameters
	----------
	arr:array
		Numpy array.

	Returns
	-------
	Image
		PIL Image.

	"""
	return Image.fromarray(arr.astype('uint8'), 'RGB')

def blend(arr1, arr2, alpha=0.5):
	"""Blend 2 arrays together, mixing with alpha.

	Parameters
	----------
	arr1:array
		Image 1.
	arr2:array
		Image 2.
	alpha:float
		Higher alpha makes image more like image 1.

	Returns
	-------
	array
		Resulting image.

	"""
	return alpha*arr1 + (1.-alpha)*arr2

def prob2rbg(prob, palette, arr):
	"""Convert probability score to rgb image.

	Parameters
	----------
	prob:float
		Between 0 and 1 score.
	palette:palette
		Pallet converts between prob and color.
	arr:array
		Original array.

	Returns
	-------
	array
		New image colored by prediction score.

	"""
	col = palette(prob)
	for i in range(3):
		arr[...,i] = int(col[i]*255)
	return arr

def seg2rgb(seg, palette, n_segmentation_classes):
	"""Color each pixel by segmentation class.

	Parameters
	----------
	seg:array
		Segmentation mask.
	palette:palette
		Color to RGB map.
	n_segmentation_classes:int
		Total number segmentation classes.

	Returns
	-------
	array
		Returned segmentation image.
	"""
	#print(seg.shape)
	#print((seg/n_segmentation_classes))
	img=(palette(seg/n_segmentation_classes)[...,:3]*255).astype(int)
	#print(img.shape)
	return img

def annotation2rgb(i,palette,arr):
	"""Go from annotation of patch to color.

	Parameters
	----------
	i:int
		Annotation index.
	palette:palette
		Index to color mapping.
	arr:array
		Image array.

	Returns
	-------
	array
		Resulting image.

	"""
	col = palette[i]
	for i in range(3):
		arr[...,i] = int(col[i]*255)
	return arr

def plot_image_(image_file, compression_factor=2., test_image_name='test.png'):
	"""Plots entire SVS/other image.

	Parameters
	----------
	image_file:str
		Image file.
	compression_factor:float
		Amount to shrink each dimension of image.
	test_image_name:str
		Output image file.

	"""
	from pathflowai.utils import svs2dask_array, npy2da
	import cv2
	arr=svs2dask_array(image_file, tile_size=1000, overlap=0, remove_last=True, allow_unknown_chunksizes=False) if (not image_file.endswith('.npy')) else npy2da(image_file)
	arr2=to_pil(cv2.resize(arr.compute(), dsize=tuple((np.array(arr.shape[:2])/compression_factor).astype(int).tolist()), interpolation=cv2.INTER_CUBIC))
	arr2.save(test_image_name)

# for now binary output
class PredictionPlotter:
	"""Plots predictions over entire image.

	Parameters
	----------
	dask_arr_dict:dict
		Stores all dask arrays corresponding to all of the images.
	patch_info_db:str
		Patch level information, eg. prediction.
	compression_factor:float
		How much to compress image by.
	alpha:float
		Low value assigns higher weight to prediction over original image.
	patch_size:int
		Patch size.
	no_db:bool
		Don't use patch information.
	plot_annotation:bool
		Plot annotations from patch information.
	segmentation:bool
		Plot segmentation mask.
	n_segmentation_classes:int
		Number segmentation classes.
	input_dir:str
		Input directory.
	annotation_col:str
		Annotation column to plot.
	scaling_factor:float
		Multiplies the prediction scores to make them appear darker on the images when predicting.
	"""
	# some patches have been filtered out, not one to one!!! figure out
	def __init__(self, dask_arr_dict, patch_info_db, compression_factor=3, alpha=0.5, patch_size=224, no_db=False, plot_annotation=False, segmentation=False, n_segmentation_classes=4, input_dir='', annotation_col='annotation', scaling_factor=1.):

		self.segmentation = segmentation
		self.scaling_factor=scaling_factor
		self.segmentation_maps = None
		self.n_segmentation_classes=float(n_segmentation_classes)
		self.pred_palette = sns.cubehelix_palette(start=0,as_cmap=True)
		if not no_db:
			self.compression_factor=compression_factor
			self.alpha = alpha
			self.patch_size = patch_size
			conn = sqlite3.connect(patch_info_db)
			patch_info=pd.read_sql('select * from "{}";'.format(patch_size),con=conn)
			conn.close()
			self.annotations = {str(a):i for i,a in enumerate(patch_info['annotation'].unique().tolist())}
			self.plot_annotation=plot_annotation
			self.palette=sns.color_palette(n_colors=len(list(self.annotations.keys())))
			#print(self.palette)
			if 'y_pred' not in patch_info.columns:
				patch_info['y_pred'] = 0.
			self.patch_info=patch_info[['ID','x','y','patch_size','annotation',annotation_col]] # y_pred
			if 0:
				for ID in predictions:
					patch_info.loc[patch_info["ID"]==ID,'y_pred'] = predictions[ID]
			self.patch_info = self.patch_info[np.isin(self.patch_info['ID'],np.array(list(dask_arr_dict.keys())))]
		if self.segmentation:
			self.segmentation_maps = {slide:da.from_array(np.load(join(input_dir,'{}_mask.npy'.format(slide)),mmap_mode='r+')) for slide in dask_arr_dict.keys()}
		#self.patch_info[['x','y','patch_size']]/=self.compression_factor
		self.dask_arr_dict = {k:v[...,:3] for k,v in dask_arr_dict.items()}

	def add_custom_segmentation(self, basename, npy):
		"""Replace segmentation mask with new custom segmentation.

		Parameters
		----------
		basename:str
			Patient ID
		npy:str
			Numpy mask.
		"""
		self.segmentation_maps[basename] = da.from_array(np.load(npy,mmap_mode='r+'))

	def generate_image(self, ID):
		"""Generate the image array for the whole slide image with predictions overlaid.

		Parameters
		----------
		ID:str
			patient ID.

		Returns
		-------
		array
			Resulting overlaid whole slide image.

		"""
		patch_info = self.patch_info[self.patch_info['ID']==ID]
		dask_arr = self.dask_arr_dict[ID]
		arr_shape = np.array(dask_arr.shape).astype(float)

		#image=da.zeros_like(dask_arr)

		arr_shape[:2]/=self.compression_factor

		arr_shape=arr_shape.astype(int).tolist()

		img = Image.new('RGB',arr_shape[:2],'white')

		for i in range(patch_info.shape[0]):
			ID,x,y,patch_size,annotation,pred = patch_info.iloc[i].tolist()
			#print(x,y,annotation)
			x_new,y_new = int(x/self.compression_factor),int(y/self.compression_factor)
			image = np.zeros((patch_size,patch_size,3))
			if self.segmentation:
				image=seg2rgb(self.segmentation_maps[ID][x:x+patch_size,y:y+patch_size].compute(),self.pred_palette, self.n_segmentation_classes)
			else:
				image=prob2rbg(pred*self.scaling_factor, self.pred_palette, image) if not self.plot_annotation else annotation2rgb(self.annotations[str(pred)],self.palette,image) # annotation
			arr=dask_arr[x:x+patch_size,y:y+patch_size].compute()
			#print(image.shape)
			blended_patch=blend(arr,image, self.alpha).transpose((1,0,2))
			blended_patch_pil = to_pil(blended_patch)
			patch_size/=self.compression_factor
			patch_size=int(patch_size)
			blended_patch_pil=blended_patch_pil.resize((patch_size,patch_size))
			img.paste(blended_patch_pil, box=(x_new,y_new), mask=None)
		return img

	def return_patch(self, ID, x, y, patch_size):
		"""Return one single patch instead of entire image.

		Parameters
		----------
		ID:str
			Patient ID
		x:int
			X coordinate.
		y:int
			Y coordinate.
		patch_size:int
			Patch size.

		Returns
		-------
		array
			Image.
		"""
		img=(self.dask_arr_dict[ID][x:x+patch_size,y:y+patch_size].compute() if not self.segmentation else seg2rgb(self.segmentation_maps[ID][x:x+patch_size,y:y+patch_size].compute(),self.pred_palette, self.n_segmentation_classes))
		return to_pil(img)

	def output_image(self, img, filename, tif=False):
		"""Output calculated image to file.

		Parameters
		----------
		img:array
			Image.
		filename:str
			Output file name.
		tif:bool
			Store in TIF format?
		"""
		if tif:
			from tifffile import imwrite
			imwrite(filename, np.array(img), photometric='rgb')
		else:
			img.save(filename)

def plot_shap(model, dataset_opts, transform_opts, batch_size, outputfilename, n_outputs=1, method='deep', local_smoothing=0.0, n_samples=20, pred_out=False):
	"""Plot shapley attributions overlaid on images for classification tasks.

	Parameters
	----------
	model:nn.Module
		Pytorch model.
	dataset_opts:dict
		Options used to configure dataset
	transform_opts:dict
		Options used to configure transformers.
	batch_size:int
		Batch size for training.
	outputfilename:str
		Output filename.
	n_outputs:int
		Number of top outputs.
	method:str
		Gradient or deep explainer.
	local_smoothing:float
		How much to smooth shapley map.
	n_samples:int
		Number shapley samples to draw.
	pred_out:bool
		Label images with binary prediction score?

	"""
	import torch
	from torch.nn import functional as F
	import numpy as np
	from torch.utils.data import DataLoader
	import shap
	from pathflowai.datasets import DynamicImageDataset
	import matplotlib
	from matplotlib import pyplot as plt
	from pathflowai.sampler import ImbalancedDatasetSampler

	out_transform=dict(sigmoid=F.sigmoid,softmax=F.softmax,none=lambda x: x)
	binary_threshold=dataset_opts.pop('binary_threshold')
	num_targets=dataset_opts.pop('num_targets')

	dataset = DynamicImageDataset(**dataset_opts)

	if dataset_opts['classify_annotations']:
		binarizer=dataset.binarize_annotations(num_targets=num_targets,binary_threshold=binary_threshold)
		num_targets=len(dataset.targets)

	dataloader_val = DataLoader(dataset,batch_size=batch_size, num_workers=10, shuffle=True if num_targets>1 else False, sampler=ImbalancedDatasetSampler(dataset) if num_targets==1 else None)
	#dataloader_test = DataLoader(dataset,batch_size=batch_size,num_workers=10, shuffle=False)

	background,y_background=next(iter(dataloader_val))
	if method=='gradient':
		background=torch.cat([background,next(iter(dataloader_val))[0]],0)
	X_test,y_test=next(iter(dataloader_val))

	if torch.cuda.is_available():
		background=background.cuda()
		X_test=X_test.cuda()

	if pred_out!='none':
		if torch.cuda.is_available():
			model2=model.cuda()
		y_test=out_transform[pred_out](model2(X_test)).detach().cpu()

	y_test=y_test.numpy()

	if method=='deep':
		e = shap.DeepExplainer(model, background)
		s=e.shap_values(X_test, ranked_outputs=n_outputs)
	elif method=='gradient':
		e = shap.GradientExplainer(model, background, batch_size=batch_size, local_smoothing=local_smoothing)
		s=e.shap_values(X_test, ranked_outputs=n_outputs, nsamples=n_samples)

	if y_test.shape[1]>1:
		y_test=y_test.argmax(axis=1)

	if n_outputs>1:
		shap_values, idx = s
	else:
		shap_values, idx = s, y_test

	#print(shap_values) # .detach().cpu()

	if num_targets == 1:
		shap_numpy = [np.swapaxes(np.swapaxes(shap_values, 1, -1), 1, 2)]
	else:
		shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
		#print(shap_numpy.shape)
	X_test_numpy=X_test.detach().cpu().numpy()
	X_test_numpy=X_test_numpy.transpose((0,2,3,1))
	for i in range(X_test_numpy.shape[0]):
		X_test_numpy[i,...]*=np.array(transform_opts['std'])
		X_test_numpy[i,...]+=np.array(transform_opts['mean'])
	X_test_numpy=X_test_numpy.transpose((0,3,1,2))
	test_numpy = np.swapaxes(np.swapaxes(X_test_numpy, 1, -1), 1, 2)
	if pred_out!='none':
		labels=y_test.astype(str)
	else:
		labels = np.array([[(dataloader_val.dataset.targets[i[j]] if num_targets>1 else str(i)) for j in range(n_outputs)] for i in idx])#[:,np.newaxis] # y_test
	if 0 and (len(labels.shape)<2 or labels.shape[1]==1):
		labels=labels.flatten()#[:np.newaxis]

	#print(labels.shape,shap_numpy.shape[0])
	plt.figure()
	shap.image_plot(shap_numpy, test_numpy, labels)# if num_targets!=1 else shap_values -test_numpy , labels=dataloader_test.dataset.targets)
	plt.savefig(outputfilename, dpi=300)

def plot_umap_images(dask_arr_dict, embeddings_file, ID=None, cval=1., image_res=300., outputfname='output_embedding.png', mpl_scatter=True, remove_background_annotation='', max_background_area=0.01, zoom=0.05, n_neighbors=10, sort_col='', sort_mode='asc'):
	"""Make UMAP embedding plot, overlaid with images.

	Parameters
	----------
	dask_arr_dict:dict
		Stored dask arrays for each WSI.
	embeddings_file:str
		Embeddings pickle file stored from running using after trainign the model.
	ID:str
		Patient ID.
	cval:float
		Deprecated
	image_res:float
		Image resolution.
	outputfname:str
		Output image file.
	mpl_scatter:bool
		Recommended: Use matplotlib for scatter plot.
	remove_background_annotation:str
		Remove the background annotations. Enter for annotation to remove.
	max_background_area:float
		Maximum backgrund area in each tile for inclusion.
	zoom:float
		How much to zoom in on each patch, less than 1 is zoom out.
	n_neighbors:int
		Number of neighbors for UMAP embedding.
	sort_col:str
		Patch info column to sort on.
	sort_mode:str
		Sort ascending or descending.

	Returns
	-------
	type
		Description of returned object.

	Inspired by: https://gist.github.com/lukemetz/be6123c7ee3b366e333a
	WIP!! Needs testing."""
	import torch
	import dask
	from dask.distributed import Client
	from umap import UMAP
	from pathflowai.visualize import PlotlyPlot
	import pandas as pd, numpy as np
	import skimage.io
	from skimage.transform import resize
	import matplotlib
	matplotlib.use('Agg')
	from matplotlib import pyplot as plt
	sns.set(style='white')

	def min_resize(img, size):
		"""
		Resize an image so that it is size along the minimum spatial dimension.
		"""
		w, h = map(float, img.shape[:2])
		if min([w, h]) != size:
			if w <= h:
				img = resize(img, (int(round((h/w)*size)), int(size)))
			else:
				img = resize(img, (int(size), int(round((w/h)*size))))
		return img

	#dask_arr = dask_arr_dict[ID]

	embeddings_dict=torch.load(embeddings_file)
	embeddings=embeddings_dict['embeddings']
	patch_info=embeddings_dict['patch_info']
	if sort_col:
		idx=np.argsort(patch_info[sort_col].values)
		if sort_mode == 'desc':
			idx=idx[::-1]
		patch_info = patch_info.iloc[idx]
		embeddings=embeddings.iloc[idx]
	if ID:
		removal_bool=(patch_info['ID']==ID).values
		patch_info = patch_info.loc[removal_bool]
		embeddings=embeddings.loc[removal_bool]
	if remove_background_annotation:
		removal_bool=(patch_info[remove_background_annotation]<=(1.-max_background_area)).values
		patch_info=patch_info.loc[removal_bool]
		embeddings=embeddings.loc[removal_bool]

	umap=UMAP(n_components=2,n_neighbors=n_neighbors)
	t_data=pd.DataFrame(umap.fit_transform(embeddings.iloc[:,:-1].values),columns=['x','y'],index=embeddings.index)

	images=[]

	for i in range(patch_info.shape[0]):
		ID=patch_info.iloc[i]['ID']
		x,y,patch_size=patch_info.iloc[i][['x','y','patch_size']].values.tolist()
		arr=dask_arr_dict[ID][x:x+patch_size,y:y+patch_size]#.transpose((2,0,1))
		images.append(arr)

	c=Client()
	images=dask.compute(images)
	c.close()

	if mpl_scatter:
		from matplotlib.offsetbox import OffsetImage, AnnotationBbox
		def imscatter(x, y, ax, imageData, zoom):
			images = []
			for i in range(len(x)):
				x0, y0 = x[i], y[i]
				img = imageData[i]
				#print(img.shape)
				image = OffsetImage(img, zoom=zoom)
				ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
				images.append(ax.add_artist(ab))

			ax.update_datalim(np.column_stack([x, y]))
			ax.autoscale()

		fig, ax = plt.subplots()
		imscatter(t_data['x'].values, t_data['y'].values, imageData=images[0], ax=ax, zoom=zoom)
		sns.despine()
		plt.savefig(outputfname,dpi=300)


	else:
		xx=t_data.iloc[:,0]
		yy=t_data.iloc[:,1]

		images = [min_resize(image, img_res) for image in images]
		max_width = max([image.shape[0] for image in images])
		max_height = max([image.shape[1] for image in images])

		x_min, x_max = xx.min(), xx.max()
		y_min, y_max = yy.min(), yy.max()
		# Fix the ratios
		sx = (x_max-x_min)
		sy = (y_max-y_min)
		if sx > sy:
			res_x = sx/float(sy)*res
			res_y = res
		else:
			res_x = res
			res_y = sy/float(sx)*res

		canvas = np.ones((res_x+max_width, res_y+max_height, 3))*cval
		x_coords = np.linspace(x_min, x_max, res_x)
		y_coords = np.linspace(y_min, y_max, res_y)
		for x, y, image in zip(xx, yy, images):
			w, h = image.shape[:2]
			x_idx = np.argmin((x - x_coords)**2)
			y_idx = np.argmin((y - y_coords)**2)
			canvas[x_idx:x_idx+w, y_idx:y_idx+h] = image

		skimage.io.imsave(outputfname, canvas)
