import plotly.graph_objs as go
import plotly.offline as py
import pandas as pd, numpy as np
import networkx as nx
import dask.array as da
from PIL import Image
import matplotlib, matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
sns.set()

class PlotlyPlot:
	def __init__(self):
		self.plots=[]

	def add_plot(self, t_data_df, G=None, color_col='color', name_col='name', xyz_cols=['x','y','z'], size=2, opacity=1.0):
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
			c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, len(colors) + 2)]
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
		if axes_off:
			fig = go.Figure(data=self.plots,layout=go.Layout(scene=dict(xaxis=dict(title='',autorange=True,showgrid=False,zeroline=False,showline=False,ticks='',showticklabels=False),
				yaxis=dict(title='',autorange=True,showgrid=False,zeroline=False,showline=False,ticks='',showticklabels=False),
				zaxis=dict(title='',autorange=True,showgrid=False,zeroline=False,showline=False,ticks='',showticklabels=False))))
		else:
			fig = go.Figure(data=self.plots)
		py.plot(fig, filename=output_fname, auto_open=False)

def to_pil(arr):
	return Image.fromarray(arr.astype('uint8'), 'RGB')

def blend(arr1, arr2, alpha=0.5):
	return alpha*arr1 + (1.-alpha)*arr2

def prob2rbg(prob, palette, arr):
	col = palette(prob)
	for i in range(3):
		arr[...,i] = int(col[i]*255)
	return arr

def annotation2rgb(i,palette,arr):
	col = palette[i]
	for i in range(3):
		arr[...,i] = int(col[i]*255)
	return arr

def plot_svs_image(svs_file, compression_factor=2., test_image_name='test.png'):
	from utils import svs2dask_array
	import cv2
	arr=svs2dask_array(svs_file, tile_size=1000, overlap=0, remove_last=True, allow_unknown_chunksizes=False)
	arr2=to_pil(cv2.resize(arr.compute(), dsize=tuple((np.array(arr.shape[:2])/compression_factor).astype(int).tolist()), interpolation=cv2.INTER_CUBIC))
	arr2.save(test_image_name)

# for now binary output
class PredictionPlotter:
	# some patches have been filtered out, not one to one!!! figure out
	def __init__(self, dask_arr_dict, patch_info_db, compression_factor=3, alpha=0.5, patch_size=224, no_db=False, plot_annotation=False):
		if not no_db:
			self.palette = sns.cubehelix_palette(start=0,as_cmap=True)
			self.compression_factor=compression_factor
			self.alpha = alpha
			self.patch_size = patch_size
			conn = sqlite3.connect(patch_info_db)
			patch_info=pd.read_sql('select * from "{}";'.format(patch_size),con=conn)
			conn.close()
			self.annotations = {a:i for i,a in enumerate(patch_info['annotation'].unique().tolist())}
			self.plot_annotation=plot_annotation
			self.palette=sns.color_palette(n_colors=len(list(self.annotations.keys())))
			if 'y_pred' not in patch_info.columns:
				patch_info['y_pred'] = 0.
			self.patch_info=patch_info[['ID','x','y','patch_size','annotation','y_pred']]
			if 0:
				for ID in predictions:
					patch_info.loc[patch_info["ID"]==ID,'y_pred'] = predictions[ID]

			self.patch_info = self.patch_info[np.isin(self.patch_info['ID'],np.array(list(dask_arr_dict.keys())))]
		#self.patch_info[['x','y','patch_size']]/=self.compression_factor
		self.dask_arr_dict = {k:v[...,:3] for k,v in dask_arr_dict.items()}

	def generate_image(self, ID):
		patch_info = self.patch_info[self.patch_info['ID']==ID]
		dask_arr = self.dask_arr_dict[ID]
		arr_shape = np.array(dask_arr.shape).astype(float)

		#image=da.zeros_like(dask_arr)

		arr_shape[:2]/=self.compression_factor

		arr_shape=arr_shape.astype(int).tolist()

		img = Image.new('RGB',arr_shape[:2],'white')

		for i in range(patch_info.shape[0]):
			ID,x,y,patch_size,annotation,pred = patch_info.iloc[i].tolist()
			print(x,y,annotation)
			x_new,y_new = int(x/self.compression_factor),int(y/self.compression_factor)
			image = np.zeros((patch_size,patch_size,3))
			image=prob2rbg(pred, self.palette, image) if not self.plot_annotation else annotation2rgb(self.annotations[annotation],self.palette,image)
			arr=dask_arr[x:x+patch_size,y:y+patch_size].compute()
			blended_patch=blend(arr,image, self.alpha).transpose((1,0,2))
			blended_patch_pil = to_pil(blended_patch)
			patch_size/=self.compression_factor
			patch_size=int(patch_size)
			blended_patch_pil=blended_patch_pil.resize((patch_size,patch_size))
			img.paste(blended_patch_pil, box=(x_new,y_new), mask=None)
		return img

	def return_patch(self, ID, x, y, patch_size):
		img=self.dask_arr_dict[ID][x:x+patch_size,y:y+patch_size].compute()
		return to_pil(img)

	def output_image(self, img, filename):
		img.save(filename)
