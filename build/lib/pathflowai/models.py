"""
models.py
=======================
Houses all of the PyTorch models to access and the corresponding Scikit-Learn like model trainer.
"""
from pathflowai.unet import UNet
from pathflowai.unet2 import NestedUNet
from pathflowai.unet4 import UNetSmall as UNet2
from pathflowai.fast_scnn import get_fast_scnn
import torch
import torchvision
from torchvision import models
from torchvision.models import segmentation as segmodels
from torch import nn
from torch.nn import functional as F
import pandas as pd, numpy as np
import matplotlib, matplotlib.pyplot as plt
import seaborn as sns
from pathflowai.schedulers import *
import pysnooper
from torch.autograd import Variable
import copy
from sklearn.metrics import roc_curve, confusion_matrix, classification_report, r2_score
sns.set()
from pathflowai.losses import GeneralizedDiceLoss, FocalLoss
from apex import amp
from torch.nn import functional as F
import time

class MLP(nn.Module):
	"""Multi-layer perceptron model.

	Parameters
	----------
	n_input:int
		Number input dimensions.
	hidden_topology:list
		List of hidden topology
	dropout_p:float
		Amount dropout.
	n_outputs:int
		Number outputs.
	binary:bool
		Binary output with sigmoid transform.
	softmax:bool
		Whether to apply softmax on output.

	"""
	def __init__(self, n_input, hidden_topology, dropout_p, n_outputs=1, binary=True, softmax=False):
		super(MLP,self).__init__()
		self.topology = [n_input]+hidden_topology+[n_outputs]
		layers = [nn.Linear(self.topology[i],self.topology[i+1]) for i in range(len(self.topology)-2)]
		for layer in layers:
			torch.nn.init.xavier_uniform_(layer.weight)
		self.layers = [nn.Sequential(layer,nn.LeakyReLU(),nn.Dropout(p=dropout_p)) for layer in layers]
		self.output_layer = nn.Linear(self.topology[-2],self.topology[-1])
		torch.nn.init.xavier_uniform_(self.output_layer.weight)
		if binary:
			output_transform = nn.Sigmoid()
		elif softmax:
			output_transform = nn.Softmax()
		else:
			output_transform = nn.Dropout(p=0.)
		self.layers.append(nn.Sequential(self.output_layer,output_transform))
		self.mlp = nn.Sequential(*self.layers)

class FixedSegmentationModule(nn.Module):
	"""Special model modification for segmentation tasks. Gets output from some of the models' forward loops.

	Parameters
	----------
	segnet:nn.Module
		Segmentation network
	"""
	def __init__(self, segnet):
		super(FixedSegmentationModule, self).__init__()
		self.segnet=segnet

	def forward(self, x):
		"""Forward pass.

		Parameters
		----------
		x:Tensor
			Input

		Returns
		-------
		Tensor
			Output from model.

		"""
		return self.segnet(x)['out']

def generate_model(pretrain,architecture,num_classes, add_sigmoid=True, n_hidden=100, segmentation=False):
	"""Generate a nn.Module for use.

	Parameters
	----------
	pretrain:bool
		Pretrain using ImageNet?
	architecture:str
		See model_training for list of all architectures you can train with.
	num_classes:int
		Number of classes to predict.
	add_sigmoid:type
		Add sigmoid non-linearity at end.
	n_hidden:int
		Number of hidden fully connected layers.
	segmentation:bool
		Whether segment task?

	Returns
	-------
	nn.Module
		Pytorch model.

	"""

	#architecture = 'resnet' + str(num_layers)
	model = None

	if architecture =='unet':
		model = UNet(n_channels=3, n_classes=num_classes)
	elif architecture =='unet2':
		model = UNet2(3,num_classes)
	elif architecture == 'fast_scnn':
		model = get_fast_scnn(num_classes)
	elif architecture == 'nested_unet':
		model = NestedUNet(3, num_classes)
	elif architecture.startswith('efficientnet'):
		from efficientnet_pytorch import EfficientNet
		if pretrain:
			model = EfficientNet.from_pretrained(architecture, override_params=dict(num_classes=num_classes))
		else:
			model = EfficientNet.from_name(architecture, override_params=dict(num_classes=num_classes))
		print(model)
	else:
		#for pretrained on imagenet
		model_names = [m for m in dir(models) if not m.startswith('__')]
		segmentation_model_names = [m for m in dir(segmodels) if not m.startswith('__')]
		if architecture in model_names:
			model = getattr(models, architecture)(pretrained=pretrain)
		if segmentation:
			if architecture in segmentation_model_names:
				model = getattr(segmodels, architecture)(pretrained=pretrain)
			else:
				model = UNet(n_channels=3)
			if architecture.startswith('deeplab'):
				model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
				model = FixedSegmentationModule(model)
			elif architecture.startswith('fcn'):
				model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
				model = FixedSegmentationModule(model)
		elif architecture.startswith('resnet') or architecture.startswith('inception'):
			num_ftrs = model.fc.in_features
			#linear_layer = nn.Linear(num_ftrs, num_classes)
			#torch.nn.init.xavier_uniform(linear_layer.weight)
			model.fc = MLP(num_ftrs, [1000], dropout_p=0., n_outputs=num_classes, binary=add_sigmoid, softmax=False).mlp#nn.Sequential(*([linear_layer]+([nn.Sigmoid()] if (add_sigmoid) else [])))
		elif architecture.startswith('alexnet') or architecture.startswith('vgg') or architecture.startswith('densenets'):
			num_ftrs = model.classifier[6].in_features
			#linear_layer = nn.Linear(num_ftrs, num_classes)
			#torch.nn.init.xavier_uniform(linear_layer.weight)
			model.classifier[6] = MLP(num_ftrs, [1000], dropout_p=0., n_outputs=num_classes, binary=add_sigmoid, softmax=False).mlp#nn.Sequential(*([linear_layer]+([nn.Sigmoid()] if (add_sigmoid) else [])))
	return model

#@pysnooper.snoop("dice_loss.log")
def dice_loss(logits, true, eps=1e-7):
	"""https://github.com/kevinzakka/pytorch-goodies
	Computes the Sørensen–Dice loss.

	Note that PyTorch optimizers minimize a loss. In this
	case, we would like to maximize the dice loss so we
	return the negated dice loss.

	Args:
		true: a tensor of shape [B, 1, H, W].
		logits: a tensor of shape [B, C, H, W]. Corresponds to
			the raw output or logits of the model.
		eps: added to the denominator for numerical stability.

	Returns:
		dice_loss: the Sørensen–Dice loss.
	"""
	#true=true.long()
	num_classes = logits.shape[1]
	if num_classes == 1:
		true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
		true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
		true_1_hot_f = true_1_hot[:, 0:1, :, :]
		true_1_hot_s = true_1_hot[:, 1:2, :, :]
		true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
		pos_prob = torch.sigmoid(logits)
		neg_prob = 1 - pos_prob
		probas = torch.cat([pos_prob, neg_prob], dim=1)
	else:
		true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
		#print(true_1_hot.size())
		true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
		probas = F.softmax(logits, dim=1)
	true_1_hot = true_1_hot.type(logits.type())
	dims = (0,) + tuple(range(2, true.ndimension()))
	intersection = torch.sum(probas * true_1_hot, dims)
	cardinality = torch.sum(probas + true_1_hot, dims)
	dice_loss = (2. * intersection / (cardinality + eps)).mean()
	return (1 - dice_loss)

class ModelTrainer:
	"""Trainer for the neural network model that wraps it into a scikit-learn like interface.

	Parameters
	----------
	model:nn.Module
		Deep learning pytorch model.
	n_epoch:int
		Number training epochs.
	validation_dataloader:DataLoader
		Dataloader of validation dataset.
	optimizer_opts:dict
		Options for optimizer.
	scheduler_opts:dict
		Options for learning rate scheduler.
	loss_fn:str
		String to call a particular loss function for model.
	reduction:str
		Mean or sum reduction of loss.
	num_train_batches:int
		Number of training batches for epoch.
	"""
	def __init__(self, model, n_epoch=300, validation_dataloader=None, optimizer_opts=dict(name='adam',lr=1e-3,weight_decay=1e-4), scheduler_opts=dict(scheduler='warm_restarts',lr_scheduler_decay=0.5,T_max=10,eta_min=5e-8,T_mult=2), loss_fn='ce', reduction='mean', num_train_batches=None):

		self.model = model
		optimizers = {'adam':torch.optim.Adam, 'sgd':torch.optim.SGD}
		loss_functions = {'bce':nn.BCEWithLogitsLoss(reduction=reduction), 'ce':nn.CrossEntropyLoss(reduction=reduction), 'mse':nn.MSELoss(reduction=reduction), 'nll':nn.NLLLoss(reduction=reduction), 'dice':dice_loss, 'focal':FocalLoss(num_class=2), 'gdl':GeneralizedDiceLoss(add_softmax=True)}
		loss_functions['dice+ce']=(lambda y_pred, y_true: dice_loss(y_pred,y_true)+loss_functions['ce'](y_pred,y_true))
		if 'name' not in list(optimizer_opts.keys()):
			optimizer_opts['name']='adam'
		self.optimizer = optimizers[optimizer_opts.pop('name')](self.model.parameters(),**optimizer_opts)
		self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O2')
		self.scheduler = Scheduler(optimizer=self.optimizer,opts=scheduler_opts)
		self.n_epoch = n_epoch
		self.validation_dataloader = validation_dataloader
		self.loss_fn = loss_functions[loss_fn]
		self.loss_fn_name = loss_fn
		self.bce=(self.loss_fn_name=='bce' or self.validation_dataloader.dataset.mt_bce)
		self.sigmoid = nn.Sigmoid()
		self.original_loss_fn = copy.deepcopy(loss_functions[loss_fn])
		self.num_train_batches = num_train_batches
		self.val_loss_fn = copy.deepcopy(loss_functions[loss_fn])

	def calc_loss(self, y_pred, y_true):
		"""Calculates loss supplied in init statement and modified by reweighting.

		Parameters
		----------
		y_pred:tensor
			Predictions.
		y_true:tensor
			True values.

		Returns
		-------
		loss

		"""

		return self.loss_fn(y_pred, y_true)

	def calc_val_loss(self, y_pred, y_true):
		"""Calculates loss supplied in init statement on validation set.

		Parameters
		----------
		y_pred:tensor
			Predictions.
		y_true:tensor
			True values.

		Returns
		-------
		val_loss

		"""

		return self.val_loss_fn(y_pred, y_true)

	def reset_loss_fn(self):
		"""Resets loss to original specified loss."""
		self.loss_fn = self.original_loss_fn

	def add_class_balance_loss(self, dataset):
		"""Updates loss function to handle class imbalance by weighting inverse to class appearance.

		Parameters
		----------
		dataset:DynamicImageDataset
			Dataset to balance by.

		"""
		self.class_weights = dataset.get_class_weights()
		self.original_loss_fn = copy.deepcopy(self.loss_fn)
		weight=torch.tensor(self.class_weights,dtype=torch.float)
		if torch.cuda.is_available():
			weight=weight.cuda()
		if self.loss_fn_name=='ce':
			self.loss_fn = nn.CrossEntropyLoss(weight=weight)
		elif self.loss_fn_name=='nll':
			self.loss_fn = nn.NLLLoss(weight=weight)
		else: # modify below for multi-target
			self.loss_fn = lambda y_pred,y_true: sum([self.class_weights[i]*self.original_loss_fn(y_pred[y_true==i],y_true[y_true==i]) for i in range(2) if sum(y_true==i)])

	def calc_best_confusion(self, y_pred, y_true):
		"""Calculate confusion matrix on validation set for classification/segmentation tasks, optimize threshold where positive.

		Parameters
		----------
		y_pred:array
			Predictions.
		y_true:array
			Ground truth.

		Returns
		-------
		float
			Optimized threshold to use on test set.
		dataframe
			Confusion matrix.

		"""
		fpr, tpr, thresholds = roc_curve(y_true, y_pred)
		threshold=thresholds[np.argmin(np.sum((np.array([0,1])-np.vstack((fpr, tpr)).T)**2,axis=1)**.5)]
		y_pred = (y_pred>threshold).astype(int)
		return threshold, pd.DataFrame(confusion_matrix(y_true,y_pred),index=['F','T'],columns=['-','+']).iloc[::-1,::-1].T

	def loss_backward(self,loss):
		"""Backprop using mixed precision for added speed boost.

		Parameters
		----------
		loss:loss
			Torch loss calculated.

		"""
		with amp.scale_loss(loss,self.optimizer) as scaled_loss:
			scaled_loss.backward()

	#@pysnooper.snoop('train_loop.log')
	def train_loop(self, epoch, train_dataloader):
		"""One training epoch, calculate predictions, loss, backpropagate.

		Parameters
		----------
		epoch:int
			Current epoch.
		train_dataloader:DataLoader
			Training data.

		Returns
		-------
		float
			Training loss for epoch

		"""
		self.model.train(True)
		running_loss = 0.
		n_batch = len(train_dataloader.dataset)//train_dataloader.batch_size if self.num_train_batches == None else self.num_train_batches
		for i, batch in enumerate(train_dataloader):
			starttime=time.time()
			if i == n_batch:
				break
			X = Variable(batch[0], requires_grad=True)
			y_true = Variable(batch[1])
			if train_dataloader.dataset.segmentation and self.loss_fn_name!='dice':
				y_true=y_true.squeeze(1)
			if torch.cuda.is_available():
				X = X.cuda()
				y_true=y_true.cuda()
			y_pred = self.model(X)
			#sizes=(y_pred.size(),y_true.size())
			loss = self.calc_loss(y_pred,y_true)
			train_loss=loss.item()
			running_loss += train_loss
			self.optimizer.zero_grad()
			self.loss_backward(loss)#loss.backward()
			self.optimizer.step()
			endtime=time.time()
			print("Epoch {}[{}/{}] Time:{}, Train Loss:{}".format(epoch,i,n_batch,round(endtime-starttime,3),train_loss))
		self.scheduler.step()
		running_loss/=n_batch
		return running_loss

	def val_loop(self, epoch, val_dataloader, print_val_confusion=True, save_predictions=True):
		"""Calculate loss over validation set.

		Parameters
		----------
		epoch:int
			Current epoch.
		val_dataloader:DataLoader
			Validation iterator.
		print_val_confusion:bool
			Calculate confusion matrix and plot.
		save_predictions:int
			Print validation results.

		Returns
		-------
		float
			Validation loss for epoch.
		"""
		self.model.train(False)
		n_batch = len(val_dataloader.dataset)//val_dataloader.batch_size
		running_loss = 0.
		Y = {'pred':[],'true':[]}
		with torch.no_grad():
			for i, batch in enumerate(val_dataloader):
				X = Variable(batch[0],requires_grad=False)
				y_true = Variable(batch[1])
				if val_dataloader.dataset.segmentation and self.loss_fn_name!='dice':
					y_true=y_true.squeeze(1)
				if torch.cuda.is_available():
					X = X.cuda()
					y_true=y_true.cuda()
				y_pred = self.model(X)
				if save_predictions:
					if val_dataloader.dataset.segmentation:
						Y['true'].append(torch.flatten(y_true if not val_dataloader.dataset.gdl else y_true).detach().cpu().numpy().astype(int).flatten()) # .argmax(axis=1)
						Y['pred'].append((y_pred.detach().cpu().numpy().argmax(axis=1)).astype(int).flatten())
					else:
						Y['true'].append(y_true.detach().cpu().numpy().astype(int).flatten())
						y_pred_numpy=((y_pred if not self.bce else self.sigmoid(y_pred)).detach().cpu().numpy()).astype(float)
						if len(y_pred_numpy)>1 and y_pred_numpy.shape[1]>1 and not val_dataloader.dataset.mt_bce:
							y_pred_numpy=y_pred_numpy.argmax(axis=1)
						Y['pred'].append(y_pred_numpy.flatten())
				loss = self.calc_val_loss(y_pred,y_true)
				val_loss=loss.item()
				running_loss += val_loss
				print("Epoch {}[{}/{}] Val Loss:{}".format(epoch,i,n_batch,val_loss))
		if print_val_confusion and save_predictions:
			y_pred,y_true = np.hstack(Y['pred']),np.hstack(Y['true'])
			if not val_dataloader.dataset.segmentation:
				if self.loss_fn_name in ['bce','mse'] and not val_dataloader.dataset.mt_bce:
					threshold, best_confusion = self.calc_best_confusion(y_pred,y_true)
					print("Epoch {} Val Confusion, Threshold {}:".format(epoch,threshold))
					print(best_confusion)
					y_true = y_true.astype(int)
					y_pred = (y_pred>=threshold).astype(int)
				elif val_dataloader.dataset.mt_bce:
					n_targets = len(val_dataloader.dataset.targets)
					y_pred=y_pred[y_true>0]
					y_true=y_true[y_true>0]
					y_true=y_true[np.isnan(y_pred)==False]
					y_pred=y_pred[np.isnan(y_pred)==False]
					if 0 and n_targets > 1:
						n_row=len(y_true)/n_targets
						y_pred=y_pred.reshape(int(n_row),n_targets)
						y_true=y_true.reshape(int(n_row),n_targets)
					print("Epoch {} Val Regression, R2 Score {}".format(epoch, str(r2_score(y_true, y_pred))))
			else:
				print(classification_report(y_true,y_pred))

		running_loss/=n_batch
		return running_loss

	#@pysnooper.snoop("test_loop.log")
	def test_loop(self, test_dataloader):
		"""Calculate final predictions on loss.

		Parameters
		----------
		test_dataloader:DataLoader
			Test dataset.

		Returns
		-------
		array
			Predictions or embeddings.
		"""
		#self.model.train(False) KEEP DROPOUT? and BATCH NORM??
		y_pred = []
		running_loss = 0.
		with torch.no_grad():
			for i, (X,y_test) in enumerate(test_dataloader):
				#X = Variable(batch[0],requires_grad=False)
				if torch.cuda.is_available():
					X = X.cuda()
				if test_dataloader.dataset.segmentation:
					prediction=self.model(X).detach().cpu().numpy().argmax(axis=1)
					pred_size=prediction.shape#size()
					#pred_mean=prediction[0].mean(axis=0)
					y_pred.append((prediction).astype(int))
				else:
					prediction=self.model(X)
					if (len(test_dataloader.dataset.targets)-1) or self.bce:
						prediction=self.sigmoid(prediction)
					elif test_dataloader.dataset.classify_annotations:
						prediction=F.softmax(prediction,dim=1)
					y_pred.append(prediction.detach().cpu().numpy())
		y_pred = np.concatenate(y_pred,axis=0)#torch.cat(y_pred,0)

		return y_pred

	def fit(self, train_dataloader, verbose=False, print_every=10, save_model=True, plot_training_curves=False, plot_save_file=None, print_val_confusion=True, save_val_predictions=True):
		"""Fits the segmentation or classification model to the patches, saving the model with the lowest validation score.

		Parameters
		----------
		train_dataloader:DataLoader
			Training dataset.
		verbose:bool
			Print training and validation loss?
		print_every:int
			Number of epochs until print?
		save_model:bool
			Whether to save model when reaching lowest validation loss.
		plot_training_curves:bool
			Plot training curves over epochs.
		plot_save_file:str
			File to save training curves.
		print_val_confusion:bool
			Print validation confusion matrix.
		save_val_predictions:bool
			Print validation results.

		Returns
		-------
		self
			Trainer.
		float
			Minimum val loss.
		int
			Best validation epoch with lowest loss.

		"""
		# choose model with best f1
		self.train_losses = []
		self.val_losses = []
		for epoch in range(self.n_epoch):
			start_time=time.time()
			train_loss = self.train_loop(epoch,train_dataloader)
			current_time=time.time()
			train_time=current_time-start_time
			self.train_losses.append(train_loss)
			val_loss = self.val_loop(epoch,self.validation_dataloader, print_val_confusion=print_val_confusion, save_predictions=save_val_predictions)
			val_time=time.time()-current_time
			self.val_losses.append(val_loss)
			if verbose and not (epoch % print_every):
				if plot_training_curves:
					self.plot_train_val_curves(plot_save_file)
				print("Epoch {}: Train Loss {}, Val Loss {}, Train Time {}, Val Time {}".format(epoch,train_loss,val_loss,train_time,val_time))
			if val_loss <= min(self.val_losses) and save_model:
				min_val_loss = val_loss
				best_epoch = epoch
				best_model = copy.deepcopy(self.model)
		if save_model:
			self.model = best_model
		return self, min_val_loss, best_epoch

	def plot_train_val_curves(self, save_file=None):
		"""Plots training and validation curves.

		Parameters
		----------
		save_file:str
			File to save to.

		"""
		plt.figure()
		sns.lineplot('epoch','value',hue='variable',
					 data=pd.DataFrame(np.vstack((np.arange(len(self.train_losses)),self.train_losses,self.val_losses)).T,
									   columns=['epoch','train','val']).melt(id_vars=['epoch'],value_vars=['train','val']))
		if save_file is not None:
			plt.savefig(save_file, dpi=300)

	def predict(self, test_dataloader):
		"""Make classification segmentation predictions on testing data.

		Parameters
		----------
		test_dataloader:DataLoader
			Test data.

		Returns
		-------
		array
			Predictions.

		"""
		y_pred = self.test_loop(test_dataloader)
		return y_pred

	def fit_predict(self, train_dataloader, test_dataloader):
		"""Fit model to training data and make classification segmentation predictions on testing data.

		Parameters
		----------
		train_dataloader:DataLoader
			Train data.
		test_dataloader:DataLoader
			Test data.

		Returns
		-------
		array
			Predictions.

		"""
		return self.fit(train_dataloader)[0].predict(test_dataloader)

	def return_model(self):
		"""Returns pytorch model.
		"""
		return self.model
