from unet import UNet
import torch
import torchvision
from torchvision import models
from torchvision.models import segmentation as segmodels
from torch import nn
import pandas as pd, numpy as np
import matplotlib, matplotlib.pyplot as plt
import seaborn as sns
from schedulers import *
import pysnooper
from torch.autograd import Variable
import copy
from sklearn.metrics import roc_curve, confusion_matrix, classification_report
sns.set()

def generate_model(pretrain,architecture,num_classes, add_sigmoid=True, n_hidden=100, segmentation=False):

	#architecture = 'resnet' + str(num_layers)
	model = None

	if architecture =='unet':
		model = UNet(n_channels=3, n_classes=num_classes)
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
			elif architecture.startswith('fcn'):
				model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
		elif architecture.startswith('resnet') or architecture.startswith('inception'):
			num_ftrs = model.fc.in_features
			linear_layer = nn.Linear(num_ftrs, num_classes)
			torch.nn.init.xavier_uniform(linear_layer.weight)
			model.fc = nn.Sequential(*([linear_layer]+([nn.Sigmoid()] if (add_sigmoid) else [])))
		elif architecture.startswith('alexnet') or architecture.startswith('vgg') or architecture.startswith('densenets'):
			num_ftrs = model.classifier[6].in_features
			linear_layer = nn.Linear(num_ftrs, num_classes)
			torch.nn.init.xavier_uniform(linear_layer.weight)
			model.classifier[6] = nn.Sequential(*([linear_layer]+([nn.Sigmoid()] if (add_sigmoid) else [])))

	return model



class ModelTrainer:
	def __init__(self, model, n_epoch=300, validation_dataloader=None, optimizer_opts=dict(name='adam',lr=1e-3,weight_decay=1e-4), scheduler_opts=dict(scheduler='warm_restarts',lr_scheduler_decay=0.5,T_max=10,eta_min=5e-8,T_mult=2), loss_fn='ce', reduction='mean'):
		self.model = model
		optimizers = {'adam':torch.optim.Adam, 'sgd':torch.optim.SGD}
		loss_functions = {'bce':nn.BCELoss(reduction=reduction), 'ce':nn.CrossEntropyLoss(reduction=reduction), 'mse':nn.MSELoss(reduction=reduction)}
		if 'name' not in list(optimizer_opts.keys()):
			optimizer_opts['name']='adam'
		self.optimizer = optimizers[optimizer_opts.pop('name')](self.model.parameters(),**optimizer_opts)
		self.scheduler = Scheduler(optimizer=self.optimizer,opts=scheduler_opts)
		self.n_epoch = n_epoch
		self.validation_dataloader = validation_dataloader
		self.loss_fn = loss_functions[loss_fn]
		self.loss_fn_name = loss_fn
		self.original_loss_fn = copy.deepcopy(loss_functions[loss_fn])

	def calc_loss(self, y_pred, y_true):
		return self.loss_fn(y_pred, y_true)

	def calc_val_loss(self, y_pred, y_true):
		return self.original_loss_fn(y_pred, y_true)

	def reset_loss_fn(self):
		self.loss_fn = self.original_loss_fn

	def add_class_balance_loss(self, dataset):
		self.class_weights = dataset.get_class_weights()
		self.original_loss_fn = copy.deepcopy(self.loss_fn)
		self.loss_fn = lambda y_pred,y_true: sum([self.original_loss_fn(y_pred[y_true==i],y_true[y_true==i]) for i in range(2)])

	def calc_best_confusion(self, y_pred, y_true):
		fpr, tpr, thresholds = roc_curve(y_true, y_pred)
		threshold=thresholds[np.argmin(np.sum((np.array([0,1])-np.vstack((fpr, tpr)).T)**2,axis=1)**.5)]
		y_pred = (y_pred>threshold).astype(int)
		return threshold, pd.DataFrame(confusion_matrix(y_true,y_pred),index=['F','T'],columns=['-','+']).iloc[::-1,::-1].T

	@pysnooper.snoop('train_loop.log')
	def train_loop(self, epoch, train_dataloader):
		self.model.train(True)
		running_loss = 0.
		n_batch = len(train_dataloader.dataset)//train_dataloader.batch_size
		for i, batch in enumerate(train_dataloader):
			X = Variable(batch[0], requires_grad=True)
			y_true = Variable(batch[1])
			if train_dataloader.dataset.segmentation:
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
			loss.backward()
			self.optimizer.step()
			print("Epoch {}[{}/{}] Train Loss:{}".format(epoch,i,n_batch,train_loss))
		self.scheduler.step()
		running_loss/=n_batch
		return running_loss

	def val_loop(self, epoch, val_dataloader, print_val_confusion=True, save_predictions=True):
		self.model.train(False)
		n_batch = len(val_dataloader.dataset)//val_dataloader.batch_size
		running_loss = 0.
		Y = {'pred':[],'true':[]}
		with torch.no_grad():
			for i, batch in enumerate(val_dataloader):
				X = Variable(batch[0],requires_grad=False)
				y_true = Variable(batch[1])
				if val_dataloader.dataset.segmentation:
					y_true=y_true.squeeze(1)
				if torch.cuda.is_available():
					X = X.cuda()
					y_true=y_true.cuda()
				y_pred = self.model(X)
				if save_predictions:
					Y['true'].append(y_true.detach().cpu().numpy().astype(int).flatten())
					Y['pred'].append((y_pred.detach().cpu().numpy()).astype(float).flatten())
				loss = self.calc_val_loss(y_pred,y_true)
				val_loss=loss.item()
				running_loss += val_loss
				print("Epoch {}[{}/{}] Val Loss:{}".format(epoch,i,n_batch,val_loss))
		if print_val_confusion and save_predictions and self.loss_fn_name!='ce':
			y_pred,y_true = np.hstack(Y['pred']),np.hstack(Y['true'])
			threshold, best_confusion = self.calc_best_confusion(y_pred,y_true)
			print("Epoch {} Val Confusion, Threshold {}:".format(epoch,threshold))
			print(best_confusion)
			print(classification_report(y_true.astype(int),(y_pred>=threshold).astype(int)))
		running_loss/=n_batch
		return running_loss

	def test_loop(self, test_dataloader):
		self.model.train(False)
		y_pred = []
		running_loss = 0.
		with torch.no_grad():
			for i, batch in enumerate(test_dataloader):
				X = Variable(batch[0],requires_grad=False)
				if torch.cuda.is_available():
					X = X.cuda()
				y_pred.append(self.model(X).detach().cpu().numpy())
		y_pred = np.concatenate(y_pred,axis=0)#torch.cat(y_pred,0)
		return y_pred

	def fit(self, train_dataloader, verbose=False, print_every=10, save_model=True, plot_training_curves=False, plot_save_file=None, print_val_confusion=True, save_val_predictions=True):
		# choose model with best f1
		self.train_losses = []
		self.val_losses = []
		for epoch in range(self.n_epoch):
			train_loss = self.train_loop(epoch,train_dataloader)
			self.train_losses.append(train_loss)
			val_loss = self.val_loop(epoch,self.validation_dataloader, print_val_confusion=print_val_confusion, save_predictions=save_val_predictions)
			self.val_losses.append(val_loss)
			if verbose and not (epoch % print_every):
				if plot_training_curves:
					self.plot_train_val_curves(plot_save_file)
				print("Epoch {}: Train Loss {}, Val Loss {}".format(epoch,train_loss,val_loss))
			if val_loss <= min(self.val_losses) and save_model:
				min_val_loss = val_loss
				best_epoch = epoch
				best_model = copy.deepcopy(self.model)
		if save_model:
			self.model = best_model
		self.train_losses = train_losses
		self.val_losses = val_losses
		return self, min_val_loss, best_epoch

	def plot_train_val_curves(self, save_file=None):
		plt.figure()
		sns.lineplot('epoch','value',hue='variable',
					 data=pd.DataFrame(np.vstack((np.arange(len(self.train_losses)),self.train_losses,self.val_losses)).T,
									   columns=['epoch','train','val']).melt(id_vars=['epoch'],value_vars=['train','val']))
		if save_file is not None:
			plt.savefig(save_file, dpi=300)

	def predict(self, test_dataloader):
		y_pred = self.test_loop(test_dataloader)
		return y_pred

	def fit_predict(self, train_dataloader, test_dataloader):
		return self.fit(train_dataloader)[0].predict(test_dataloader)

	def return_model(self):
		return self.model
