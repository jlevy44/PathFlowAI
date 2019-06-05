import torch
from torchvision import transforms
import os
import dask.array as da, pandas as pd, numpy as np
from utils import *
import pysnooper
import nonechucks as nc
from torch.utils.data import Dataset
import random
import albumentations as alb
import copy
from albumentations import pytorch as albtorch
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight

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


def get_normalizer(normalization_file, dataset_df, patch_info_file, input_dir, target_names, pos_annotation_class, segmentation, patch_size, fix_names, other_annotations):
	if os.path.exists(normalization_file):
		norm_dict = torch.load(normalization_file)
	else:
		norm_dict = {'normalization_file':normalization_file}

	if 'normalization_file' in norm_dict:

		transformers = get_data_transforms(patch_size = 224, mean=[], std=[], resize=True, transform_platform='torch')

		print(dict(pos_annotation_class=pos_annotation_class, segmentation=segmentation, patch_size=patch_size, fix_names=fix_names, other_annotations=other_annotations))

		dataset = nc.SafeDataset(DynamicImageDataset(dataset_df, 'pass', patch_info_file, transformers, input_dir, target_names, pos_annotation_class=pos_annotation_class, segmentation=segmentation, patch_size=patch_size, fix_names=fix_names, other_annotations=other_annotations))

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
	def __init__(self,dataset_df,set, patch_info_file, transformers, input_dir, target_names, pos_annotation_class, other_annotations=[], segmentation=False, patch_size=224, fix_names=True, target_segmentation_class=-1, target_threshold=0., oversampling_factor=1):
		self.transformer=transformers[set]
		original_set = copy.deepcopy(set)
		if set=='pass':
			set='train'
		self.targets = target_names
		if len(self.targets)==1:
			self.targets = self.targets[0]
		self.set = set
		self.segmentation = segmentation
		if original_set == 'pass':
			self.transform_fn = lambda x,y: (self.transformer(x), torch.tensor(1.,dtype=torch.float))
		else:
			if self.segmentation:
				self.transform_fn = lambda x,y: segmentation_transform(x,y, self.transformer)
			else:
				if 'p' in dir(self.transformer):
					self.transform_fn = lambda x,y: (self.transformer(True, image=x)['image'], torch.tensor(y,dtype=torch.float))
				else:
					self.transform_fn = lambda x,y: (self.transformer(x), torch.tensor(y,dtype=torch.float))
		self.image_set = dataset_df[dataset_df['set']==set]
		if self.segmentation:
			self.targets='target'
			self.image_set[self.targets] = 1.
		if not self.segmentation and fix_names:
			self.image_set.loc[:,'ID'] = self.image_set['ID'].map(fix_name)
		self.slide_info = pd.DataFrame(self.image_set.set_index('ID').loc[:,self.targets])
		IDs = self.slide_info.index.tolist()

		self.patch_info = modify_patch_info(patch_info_file, self.slide_info, pos_annotation_class, patch_size, self.segmentation, other_annotations, target_segmentation_class, target_threshold)

		if self.segmentation and original_set!='pass':
			IDs = self.patch_info['ID'].unique()
			self.segmentation_maps = {slide:da.from_array(np.load(join(input_dir,'{}_mask.npy'.format(slide)),mmap_mode='r+')) for slide in IDs}
		self.slides = {slide:da.from_zarr(join(input_dir,'{}.zarr'.format(slide))) for slide in IDs}
		#print(self.slide_info)
		if original_set =='pass':
			self.segmentation=False
		#print(self.patch_info[self.targets].unique())
		if oversampling_factor > 1:
			self.patch_info = pd.concat([self.patch_info]*oversampling_factor,axis=0)
		self.length = self.patch_info.shape[0]

	def get_class_weights(self, i=0):
		return compute_class_weight(class_weight='balanced',classes=[0,1],y=self.patch_info[self.targets if type(self.targets)==type('') else self.targets[i]])

	def binarize_annotations(self, binarizer=None):
		annotations = self.patch_info['annotation']
		if binarizer == None:
			self.binarizer = LabelBinarizer().fit(annotations)
		else:
			self.binarizer = copy.deepcopy(binarizer)
		self.targets = self.binarizer.classes_
		annotation_labels = pd.DataFrame(self.binarizer.transform(annotations),index=self.patch_info.index,columns=self.targets)
		self.patch_info = pd.concat([self.patch_info,annotation_labels],axis=1)
		return self.binarizer

	def subsample(self, p):
		np.random.seed(42)
		self.patch_info = self.patch_info.sample(frac=p)
		self.length = self.patch_info.shape[0]

	#@pysnooper.snoop('get_item.log')
	def __getitem__(self, i):
		patch_info = self.patch_info.iloc[i]
		ID = patch_info['ID']
		y = patch_info[self.targets]
		xs = patch_info['x']
		ys = patch_info['y']
		patch_size = patch_info['patch_size']
		image, y = self.transform_fn(self.slides[ID][xs:xs+patch_size,ys:ys+patch_size,:3].compute().astype(np.uint8), y if not self.segmentation else np.array(self.segmentation_maps[ID][xs:xs+patch_size,ys:ys+patch_size]))#.unsqueeze(0) # transpose .transpose([1,0,2])
		#image_size=image.size()
		return image, y

	def __len__(self):
		return self.length
