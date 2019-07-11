import torch
from torchvision import transforms
import os
import dask
#from dask.distributed import Client; Client()
import dask.array as da, pandas as pd, numpy as np
from utils import *
import pysnooper
import nonechucks as nc
from torch.utils.data import Dataset, DataLoader
import random
import albumentations as alb
import copy
from albumentations import pytorch as albtorch
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from losses import class2one_hot


def RandomRotate90():
	return (lambda img: img.rotate(random.sample([0, 90, 180, 270], k=1)[0]))

def get_data_transforms(patch_size = None, mean=[], std=[], resize=False, transform_platform='torch', elastic=True):

	data_transforms = { 'torch': {
		'train': transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((patch_size,patch_size)),
			transforms.CenterCrop(patch_size),   # if not resize else
			transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
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
		'unnormalize': transforms.Compose([
			transforms.ToPILImage(),
			transforms.Normalize([1/0.15, 1/0.15, 1/0.15], [1/0.15, 1/0.15, 1/0.15])
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
		alb.augmentations.transforms.CenterCrop(patch_size, patch_size)
		]+([alb.augmentations.transforms.Flip(p=0.5),
		alb.augmentations.transforms.Transpose(p=0.5),
		alb.augmentations.transforms.ShiftScaleRotate(p=0.5)] if not elastic else [alb.augmentations.transforms.RandomRotate90(p=0.5),
		alb.augmentations.transforms.ElasticTransform(p=0.5)])+[albtorch.transforms.ToTensor(normalize=dict(mean=mean if mean else [0.7, 0.6, 0.7], std=std if std is not None else [0.15, 0.15, 0.15]))]
	),
	'val':alb.core.composition.Compose([
		alb.augmentations.transforms.Resize(patch_size, patch_size),
		alb.augmentations.transforms.CenterCrop(patch_size, patch_size),
		albtorch.transforms.ToTensor(normalize=dict(mean=mean if mean else [0.7, 0.6, 0.7], std=std if std is not None else [0.15, 0.15, 0.15]))
	])
	}}

	return data_transforms[transform_platform]

def create_transforms(mean, std):
	return get_data_transforms(patch_size = 224, mean=mean, std=std, resize=True)



def get_normalizer(normalization_file, dataset_opts):
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

def segmentation_transform(img,mask, transformer):
	res=transformer(True, image=img, mask=mask)
	#res_mask_shape = res['mask'].size()
	return res['image'], res['mask'].long()#.view(res_mask_shape[0],res_mask_shape[1],res_mask_shape[2])

class DynamicImageDataset(Dataset): # when building transformers, need a resize patch size to make patches 224 by 224
	#@pysnooper.snoop('init_data.log')
	def __init__(self,dataset_df, set, patch_info_file, transformers, input_dir, target_names, pos_annotation_class, other_annotations=[], segmentation=False, patch_size=224, fix_names=True, target_segmentation_class=-1, target_threshold=0., oversampling_factor=1, n_segmentation_classes=4, gdl=False, mt_bce=False, classify_annotations=False):
		self.transformer=transformers[set]
		original_set = copy.deepcopy(set)
		if set=='pass':
			set='train'
		self.targets = target_names
		self.mt_bce=mt_bce
		self.set = set
		self.segmentation = segmentation
		if len(self.targets)==1:
			self.targets = self.targets[0]
		if original_set == 'pass':
			self.transform_fn = lambda x,y: (self.transformer(x), torch.tensor(1.,dtype=torch.float))
		else:
			if self.segmentation:
				self.transform_fn = lambda x,y: segmentation_transform(x,y, self.transformer)
			else:
				if 'p' in dir(self.transformer):
					self.transform_fn = lambda x,y: (self.transformer(True, image=x)['image'], torch.from_numpy(y).float())
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
		self.patch_info = modify_patch_info(patch_info_file, self.slide_info, pos_annotation_class, patch_size, self.segmentation, other_annotations, target_segmentation_class, target_threshold, classify_annotations)

		if self.segmentation and original_set!='pass':
			#IDs = self.patch_info['ID'].unique()
			self.segmentation_maps = {slide:da.from_array(np.load(join(input_dir,'{}_mask.npy'.format(slide)),mmap_mode='r+')) for slide in IDs}
		self.slides = {slide:da.from_zarr(join(input_dir,'{}.zarr'.format(slide))) for slide in IDs}
		#print(self.slide_info)
		if original_set =='pass':
			self.segmentation=False
		#print(self.patch_info[self.targets].unique())
		if oversampling_factor > 1:
			self.patch_info = pd.concat([self.patch_info]*oversampling_factor,axis=0).reset_index(drop=True)
		self.length = self.patch_info.shape[0]
		self.n_segmentation_classes = n_segmentation_classes
		self.gdl=gdl if self.segmentation else False
		self.binarized=False
		self.classify_annotations=classify_annotations
		print(self.targets)

	def concat(self, other_dataset):
		self.patch_info = pd.concat([self.patch_info, other_dataset.patch_info],axis=0).reset_index(drop=True)
		self.length = self.patch_info.shape[0]
		if self.segmentation:
			self.segmentation_maps.update(other_dataset.segmentation_maps)
			#print(self.segmentation_maps.keys())

	def retain_ID(self, ID):
		self.patch_info=self.patch_info.loc[self.patch_info['ID']==ID]
		self.length = self.patch_info.shape[0]
		return self

	def split_by_ID(self):
		for ID in self.patch_info['ID'].unique():
			new_dataset = copy.deepcopy(self)
			yield ID, new_dataset.retain_ID(ID)

	def get_class_weights(self, i=0):#[0,1]
		if self.segmentation:
			weights=1./(self.patch_info[list(map(str,list(range(self.n_segmentation_classes))))].sum(axis=0).values)
		elif self.mt_bce:
			weights=1./(self.patch_info[self.targets].sum(axis=0).values)
			weights=weights/sum(weights)
		else:
			if self.binarized:
				y=np.argmax(self.patch_info[self.targets].values,axis=1)
			elif (type(self.targets)!=type('')):
				y=self.patch_info[self.targets]
			else:
				y=self.patch_info[self.targets[i]]
			weights=compute_class_weight(class_weight='balanced',classes=np.unique(y),y=y)
		return weights

	def binarize_annotations(self, binarizer=None):
		annotations = self.patch_info['annotation']
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
		#self.patch_info = pd.concat([self.patch_info,annotation_labels],axis=1)
		self.binarized=True
		return self.binarizer

	def subsample(self, p):
		np.random.seed(42)
		self.patch_info = self.patch_info.sample(frac=p)
		self.length = self.patch_info.shape[0]

	#@pysnooper.snoop('get_item.log')
	def __getitem__(self, i):
		patch_info = self.patch_info.iloc[i]
		ID = patch_info['ID']
		targets=self.targets
		if not self.segmentation:
			y = patch_info[self.targets]
			if isinstance(y,pd.Series):
				y=y.values.astype(float)
				if self.binarized:
					y=np.array(y.argmax())
			y=np.array(y)
			if not y.shape:
				y=y.reshape(1)
		xs = patch_info['x']
		ys = patch_info['y']
		patch_size = patch_info['patch_size']
		y=(y if not self.segmentation else np.array(self.segmentation_maps[ID][xs:xs+patch_size,ys:ys+patch_size]))
		image, y = self.transform_fn(self.slides[ID][xs:xs+patch_size,ys:ys+patch_size,:3].compute().astype(np.uint8), y)#.unsqueeze(0) # transpose .transpose([1,0,2])
		if not self.segmentation and not self.mt_bce and self.classify_annotations:
			y=y.long()
		#image_size=image.size()
		if self.gdl:
			y=class2one_hot(y, self.n_segmentation_classes)
		#	y=one_hot2dist(y)
		return image, y

	def __len__(self):
		return self.length
