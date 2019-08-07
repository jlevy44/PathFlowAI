"""
datasets.py
=======================
Houses the DynamicImageDataset class, also functions to help with image color channel normalization, transformers, etc..
"""

import torch
from torchvision import transforms
import os
import dask
#from dask.distributed import Client; Client()
import dask.array as da, pandas as pd, numpy as np
from pathflowai.utils import *
import pysnooper
import nonechucks as nc
from torch.utils.data import Dataset, DataLoader
import random
import albumentations as alb
import copy
from albumentations import pytorch as albtorch
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from pathflowai.losses import class2one_hot
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def RandomRotate90():
	"""Transformer for random 90 degree rotation image.

	Returns
	-------
	function
		Transformer function for operation.

	"""
	return (lambda img: img.rotate(random.sample([0, 90, 180, 270], k=1)[0]))

def get_data_transforms(patch_size = None, mean=[], std=[], resize=False, transform_platform='torch', elastic=True):
	"""Get data transformers for training test and validation sets.

	Parameters
	----------
	patch_size:int
		Original patch size being transformed.
	mean:list of float
		Mean RGB
	std:list of float
		Std RGB
	resize:int
		Which patch size to resize to.
	transform_platform:str
		Use pytorch or albumentation transforms.
	elastic:bool
		Whether to add elastic deformations from albumentations.

	Returns
	-------
	dict
		Transformers.

	"""

	data_transforms = { 'torch': {
		'train': transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((patch_size,patch_size)),
			transforms.CenterCrop(patch_size),   # if not resize else
			transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.5),
			transforms.RandomHorizontalFlip(),
			transforms.RandomVerticalFlip(),
			RandomRotate90(),
			transforms.ToTensor(),
			transforms.Normalize(mean if mean else [0.7, 0.6, 0.7], std if std is not None else [0.15, 0.15, 0.15]) #mean and standard deviations for lung adenocarcinoma resection slides
		]),
		'val': transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((patch_size,patch_size)),
			transforms.CenterCrop(patch_size),
			transforms.ToTensor(),
			transforms.Normalize(mean if mean else [0.7, 0.6, 0.7], std if std is not None else [0.15, 0.15, 0.15])
		]),
		'test': transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((patch_size,patch_size)),
			transforms.CenterCrop(patch_size),
			transforms.ToTensor(),
			transforms.Normalize(mean if mean else [0.7, 0.6, 0.7], std if std is not None else [0.15, 0.15, 0.15])
		]),
		'pass': transforms.Compose([
			transforms.ToPILImage(),
			transforms.CenterCrop(patch_size),
			transforms.ToTensor(),
		])
	},
	'albumentations':{
	'train':alb.core.composition.Compose([
		alb.augmentations.transforms.Resize(patch_size, patch_size),
		alb.augmentations.transforms.CenterCrop(patch_size, patch_size),
		alb.augmentations.transforms.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5)
		]+([alb.augmentations.transforms.Flip(p=0.5),
		alb.augmentations.transforms.Transpose(p=0.5),
		alb.augmentations.transforms.ShiftScaleRotate(p=0.5)] if not elastic else [alb.augmentations.transforms.RandomRotate90(p=0.5),
		alb.augmentations.transforms.ElasticTransform(p=0.5)])
	),
	'val':alb.core.composition.Compose([
		alb.augmentations.transforms.Resize(patch_size, patch_size),
		alb.augmentations.transforms.CenterCrop(patch_size, patch_size)
	]),
	'test':alb.core.composition.Compose([
		alb.augmentations.transforms.Resize(patch_size, patch_size),
		alb.augmentations.transforms.CenterCrop(patch_size, patch_size)
	]),
	'normalize':transforms.Compose([transforms.Normalize(mean if mean else [0.7, 0.6, 0.7], std if std is not None else [0.15, 0.15, 0.15])])
	}}

	return data_transforms[transform_platform]

def create_transforms(mean, std):
	"""Create transformers.

	Parameters
	----------
	mean:list
		See get_data_transforms.
	std:list
		See get_data_transforms.

	Returns
	-------
	dict
		Transformers.

	"""
	return get_data_transforms(patch_size = 224, mean=mean, std=std, resize=True)



def get_normalizer(normalization_file, dataset_opts):
	"""Find mean and standard deviation of images in batches.

	Parameters
	----------
	normalization_file:str
		File to store normalization information.
	dataset_opts:type
		Dictionary storing information to create DynamicDataset class.

	Returns
	-------
	dict
		Stores RGB mean, stdev.

	"""
	if os.path.exists(normalization_file):
		norm_dict = torch.load(normalization_file)
	else:
		norm_dict = {'normalization_file':normalization_file}

	if 'normalization_file' in norm_dict:

		transformers = get_data_transforms(patch_size = 224, mean=[], std=[], resize=True, transform_platform='torch')

		dataset_opts['transformers']=transformers
		#print(dict(pos_annotation_class=pos_annotation_class, segmentation=segmentation, patch_size=patch_size, fix_names=fix_names, other_annotations=other_annotations))

		dataset = DynamicImageDataset(**dataset_opts)#nc.SafeDataset(DynamicImageDataset(**dataset_opts))

		if dataset_opts['classify_annotations']:
			dataset.binarize_annotations()

		dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

		all_mean = torch.tensor([0.,0.,0.],dtype=torch.float)#[]

		all_std = torch.tensor([0.,0.,0.],dtype=torch.float)

		if torch.cuda.is_available():
			all_mean=all_mean.cuda()
			all_std=all_std.cuda()

		with torch.no_grad():
			for i,(X,_) in enumerate(dataloader): # x,3,224,224
				if torch.cuda.is_available():
					X=X.cuda()
				all_mean += torch.mean(X, (0,2,3))
				all_std += torch.std(X, (0,2,3))

		N=i+1

		all_mean /= float(N) #(np.array(all_mean).mean(axis=0)).tolist()
		all_std /= float(N) #(np.array(all_std).mean(axis=0)).tolist()

		all_mean = all_mean.detach().cpu().numpy().tolist()
		all_std = all_std.detach().cpu().numpy().tolist()

		torch.save(dict(mean=all_mean,std=all_std),norm_dict['normalization_file'])

		norm_dict = torch.load(norm_dict['normalization_file'])
	return norm_dict

def segmentation_transform(img,mask, transformer, normalizer, alb_reduction):
	"""Run albumentations and return an image and its segmentation mask.

	Parameters
	----------
	img:array
		Image as array
	mask:array
		Categorical pixel by pixel.
	transformer :
		Transformation object.

	Returns
	-------
	tuple arrays
		Image and mask array.

	"""
	res=transformer(True, image=img, mask=mask)
	#res_mask_shape = res['mask'].size()
	return normalizer(torch.tensor(np.transpose(res['image']/alb_reduction,axes=(2,0,1)),dtype=torch.float)).float(), torch.tensor(res['mask']).long()#.view(res_mask_shape[0],res_mask_shape[1],res_mask_shape[2])

class DynamicImageDataset(Dataset):
	"""Generate image dataset that accesses images and annotations via dask.

	Parameters
	----------
	dataset_df:dataframe
		Dataframe with WSI, which set it is in (train/test/val) and corresponding WSI labels if applicable.
	set:str
		Whether train, test, val or pass (normalization) set.
	patch_info_file:str
		SQL db with positional and annotation information on each slide.
	transformers:dict
		Contains transformers to apply on images.
	input_dir:str
		Directory where images comes from.
	target_names:list/str
		Names of initial targets, which may be modified.
	pos_annotation_class:str
		If selected and predicting on WSI, this class is labeled as a positive from the WSI, while the other classes are not.
	other_annotations:list
		Other annotations to consider from patch info db.
	segmentation:bool
		Conducting segmentation task?
	patch_size:int
		Patch size.
	fix_names:bool
		Whether to change the names of dataset_df.
	target_segmentation_class:list
		Now can be used for classification as well, matched with two below options, samples images only from this class. Can specify this and below two options multiple times.
	target_threshold:list
		Sampled only if above this threshold of occurence in the patches.
	oversampling_factor:list
		Over sample them at this amount.
	n_segmentation_classes:int
		Number classes to segment.
	gdl:bool
		Using generalized dice loss?
	mt_bce:bool
		For multi-target prediction tasks.
	classify_annotations:bool
		For classifying annotations.

	"""
	# when building transformers, need a resize patch size to make patches 224 by 224
	#@pysnooper.snoop('init_data.log')
	def __init__(self,dataset_df, set, patch_info_file, transformers, input_dir, target_names, pos_annotation_class, other_annotations=[], segmentation=False, patch_size=224, fix_names=True, target_segmentation_class=-1, target_threshold=0., oversampling_factor=1., n_segmentation_classes=4, gdl=False, mt_bce=False, classify_annotations=False):

		#print('check',classify_annotations)
		reduce_alb=True
		self.patch_size=patch_size
		self.input_dir = input_dir
		self.alb_reduction=255. if reduce_alb else 1.
		self.transformer=transformers[set]
		original_set = copy.deepcopy(set)
		if set=='pass':
			set='train'
		self.targets = target_names
		self.mt_bce=mt_bce
		self.set = set
		self.segmentation = segmentation
		self.alb_normalizer=None
		if 'normalize' in transformers:
			self.alb_normalizer = transformers['normalize']
		if len(self.targets)==1:
			self.targets = self.targets[0]
		if original_set == 'pass':
			self.transform_fn = lambda x,y: (self.transformer(x), torch.tensor(1.,dtype=torch.float))
		else:
			if self.segmentation:
				self.transform_fn = lambda x,y: segmentation_transform(x,y, self.transformer, self.alb_normalizer, self.alb_reduction)
			else:
				if 'p' in dir(self.transformer):
					self.transform_fn = lambda x,y: (self.alb_normalizer(torch.tensor(np.transpose(self.transformer(True, image=x)['image']/self.alb_reduction,axes=(2,0,1)),dtype=torch.float)), torch.from_numpy(y).float())
				else:
					self.transform_fn = lambda x,y: (self.transformer(x), torch.from_numpy(y).float())
		self.image_set = dataset_df[dataset_df['set']==set]
		if self.segmentation:
			self.targets='target'
			self.image_set[self.targets] = 1.
		if not self.segmentation and fix_names:
			self.image_set.loc[:,'ID'] = self.image_set['ID'].map(fix_name)
		self.slide_info = pd.DataFrame(self.image_set.set_index('ID').loc[:,self.targets])
		if self.mt_bce and not self.segmentation:
			if pos_annotation_class:
				self.targets = [pos_annotation_class]+list(other_annotations)
			else:
				self.targets = None
		print(self.targets)
		IDs = self.slide_info.index.tolist()
		pi_dict=dict(input_info_db=patch_info_file, slide_labels=self.slide_info, pos_annotation_class=pos_annotation_class, patch_size=patch_size, segmentation=self.segmentation, other_annotations=other_annotations, target_segmentation_class=target_segmentation_class, target_threshold=target_threshold, classify_annotations=classify_annotations)
		self.patch_info = modify_patch_info(**pi_dict)

		if self.segmentation and original_set!='pass':
			#IDs = self.patch_info['ID'].unique()
			self.segmentation_maps = {slide:da.from_array(np.load(join(input_dir,'{}_mask.npy'.format(slide)),mmap_mode='r+')) for slide in IDs}
		self.slides = {slide:da.from_zarr(join(input_dir,'{}.zarr'.format(slide))) for slide in IDs}
		#print(self.slide_info)
		if original_set =='pass':
			self.segmentation=False
		#print(self.patch_info[self.targets].unique())
		if oversampling_factor > 1:
			self.patch_info = pd.concat([self.patch_info]*int(oversampling_factor),axis=0).reset_index(drop=True)
		elif oversampling_factor < 1:
			self.patch_info = self.patch_info.sample(frac=oversampling_factor).reset_index(drop=True)
		self.length = self.patch_info.shape[0]
		self.n_segmentation_classes = n_segmentation_classes
		self.gdl=gdl if self.segmentation else False
		self.binarized=False
		self.classify_annotations=classify_annotations
		print(self.targets)

	def concat(self, other_dataset):
		"""Concatenate this dataset with others. Updates its own internal attributes.

		Parameters
		----------
		other_dataset:DynamicImageDataset
			Other image dataset.

		"""
		self.patch_info = pd.concat([self.patch_info, other_dataset.patch_info],axis=0).reset_index(drop=True)
		self.length = self.patch_info.shape[0]
		if self.segmentation:
			self.segmentation_maps.update(other_dataset.segmentation_maps)
			#print(self.segmentation_maps.keys())

	def retain_ID(self, ID):
		"""Reduce the sample set to just images from one ID.

		Parameters
		----------
		ID:str
			Basename/ID to predict on.

		Returns
		-------
		self

		"""
		self.patch_info=self.patch_info.loc[self.patch_info['ID']==ID]
		self.length = self.patch_info.shape[0]
		return self

	def split_by_ID(self):
		"""Generator similar to groupby, but splits up by ID, generates (ID,data) using retain_ID.

		Returns
		-------
		generator
			ID, DynamicDataset

		"""
		for ID in self.patch_info['ID'].unique():
			new_dataset = copy.deepcopy(self)
			yield ID, new_dataset.retain_ID(ID)

	def get_class_weights(self, i=0):#[0,1]
		"""Weight loss function with weights inversely proportional to the class appearence.

		Parameters
		----------
		i:int
			If multi-target, class used for weighting.

		Returns
		-------
		self
			Dataset.

		"""
		if self.segmentation:
			label_counts=self.patch_info[list(map(str,list(range(self.n_segmentation_classes))))].sum(axis=0).values
			freq = label_counts/sum(label_counts)
			weights=1./(freq)
		elif self.mt_bce:
			weights=1./(self.patch_info[self.targets].sum(axis=0).values)
			weights=weights/sum(weights)
		else:
			if self.binarized and len(self.targets)>1:
				y=np.argmax(self.patch_info[self.targets].values,axis=1)
			elif (type(self.targets)!=type('')):
				y=self.patch_info[self.targets]
			else:
				y=self.patch_info[self.targets[i]]
			y=y.values.astype(int).flatten()
			weights=compute_class_weight(class_weight='balanced',classes=np.unique(y),y=y)
		return weights

	def binarize_annotations(self, binarizer=None, num_targets=1, binary_threshold=0.):
		"""Label binarize some annotations or threshold them if classifying slide annotations.

		Parameters
		----------
		binarizer:LabelBinarizer
			Binarizes the labels of a column(s)
		num_targets:int
			Number of desired targets to preidict on.
		binary_threshold:float
			Amount of annotation in patch before positive annotation.

		Returns
		-------
		binarizer

		"""

		annotations = self.patch_info['annotation']
		annots=[annot for annot in list(self.patch_info.iloc[:,6:]) if annot !='area']
		if not self.mt_bce and num_targets > 1:
			if binarizer == None:
				self.binarizer = LabelBinarizer().fit(annotations)
			else:
				self.binarizer = copy.deepcopy(binarizer)
			self.targets = self.binarizer.classes_
			annotation_labels = pd.DataFrame(self.binarizer.transform(annotations),index=self.patch_info.index,columns=self.targets).astype(float)
			for col in list(annotation_labels):
				if col in list(self.patch_info):
					self.patch_info.loc[:,col]=annotation_labels[col].values
				else:
					self.patch_info[col]=annotation_labels[col].values
		else:
			self.binarizer=None
			self.targets=annots
			if num_targets == 1:
				self.targets = [self.targets[-1]]
			if binary_threshold>0.:
				self.patch_info.loc[:,self.targets]=(self.patch_info[self.targets]>=binary_threshold).values.astype(np.float32)
			print(self.targets)
			#self.patch_info = pd.concat([self.patch_info,annotation_labels],axis=1)
		self.binarized=True
		return self.binarizer

	def subsample(self, p):
		"""Sample subset of dataset.

		Parameters
		----------
		p:float
			Fraction to subsample.

		"""
		np.random.seed(42)
		self.patch_info = self.patch_info.sample(frac=p)
		self.length = self.patch_info.shape[0]

	def update_dataset(self, input_dir, new_db):
		"""Experimental. Only use for segmentation for now."""
		self.input_dir=input_dir
		self.patch_info=load_sql_df(new_db, self.patch_size)
		IDs = self.patch_info['ID'].unique()
		self.slides = {slide:da.from_zarr(join(self.input_dir,'{}.zarr'.format(slide))) for slide in IDs}
		if self.segmentation:
			self.segmentation_maps = {slide:da.from_array(np.load(join(self.input_dir,'{}_mask.npy'.format(slide)),mmap_mode='r+')) for slide in IDs}
		self.length = self.patch_info.shape[0]

	#@pysnooper.snoop('get_item.log')
	def __getitem__(self, i):
		patch_info = self.patch_info.iloc[i]
		ID = patch_info['ID']
		targets=self.targets
		use_long=False
		if not self.segmentation:
			y = patch_info[self.targets]
			if isinstance(y,pd.Series):
				y=y.values.astype(float)
				if self.binarized and not self.mt_bce and len(y)>1:
					y=np.array(y.argmax())
					use_long=True
			y=np.array(y)
			if not y.shape:
				y=y.reshape(1)
		xs = patch_info['x']
		ys = patch_info['y']
		patch_size = patch_info['patch_size']
		y=(y if not self.segmentation else np.array(self.segmentation_maps[ID][xs:xs+patch_size,ys:ys+patch_size]))
		image, y = self.transform_fn(self.slides[ID][xs:xs+patch_size,ys:ys+patch_size,:3].compute().astype(np.uint8), y)#.unsqueeze(0) # transpose .transpose([1,0,2])
		if not self.segmentation and not self.mt_bce and self.classify_annotations and use_long:
			y=y.long()
		#image_size=image.size()
		if self.gdl:
			y=class2one_hot(y, self.n_segmentation_classes)
		#	y=one_hot2dist(y)
		return image, y

	def __len__(self):
		return self.length
