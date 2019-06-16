from unet import UNet
from unet2 import NestedUNet
import torch
import torchvision
from torchvision import models
from torchvision.models import segmentation as segmodels
from torch import nn
from torch.nn import functional as F
import pandas as pd, numpy as np
import matplotlib, matplotlib.pyplot as plt
import seaborn as sns
from schedulers import *
import pysnooper
from torch.autograd import Variable
import copy
from sklearn.metrics import roc_curve, confusion_matrix, classification_report
sns.set()
from apex import amp


class FixedSegmentationModule(nn.Module):
	def __init__(self, segnet):
		super(FixedSegmentationModule, self).__init__()
		self.segnet=segnet

	def forward(self, x):
		return self.segnet(x)['out']

def generate_model(pretrain,architecture,num_classes, add_sigmoid=True, n_hidden=100, segmentation=False):

	#architecture = 'resnet' + str(num_layers)
	model = None

	if architecture =='unet':
		model = UNet(n_channels=3, n_classes=num_classes)
	elif architecture == 'nested_unet':
		model = NestedUNet(3, num_classes)
	elif architecture.startswith('efficientnet'):
		from efficientnet_pytorch import EfficientNet
		if pretrain:
			model = EfficientNet.from_pretrained(architecture, override_params=dict(num_classes=num_classes))
		else:
			model = EfficientNet.from_name(architecture, override_params=dict(num_classes=num_classes))
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
			linear_layer = nn.Linear(num_ftrs, num_classes)
			torch.nn.init.xavier_uniform(linear_layer.weight)
			model.fc = nn.Sequential(*([linear_layer]+([nn.Sigmoid()] if (add_sigmoid) else [])))
		elif architecture.startswith('alexnet') or architecture.startswith('vgg') or architecture.startswith('densenets'):
			num_ftrs = model.classifier[6].in_features
			linear_layer = nn.Linear(num_ftrs, num_classes)
			torch.nn.init.xavier_uniform(linear_layer.weight)
			model.classifier[6] = nn.Sequential(*([linear_layer]+([nn.Sigmoid()] if (add_sigmoid) else [])))
	return model

class FocalLoss(nn.Module): # add boundary loss
    """
	# https://raw.githubusercontent.com/Hsuxu/Loss_ToolBox-PyTorch/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):

        # logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(self.num_class-1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


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
	def __init__(self, model, n_epoch=300, validation_dataloader=None, optimizer_opts=dict(name='adam',lr=1e-3,weight_decay=1e-4), scheduler_opts=dict(scheduler='warm_restarts',lr_scheduler_decay=0.5,T_max=10,eta_min=5e-8,T_mult=2), loss_fn='ce', reduction='mean', num_train_batches=None):
		self.model = model
		optimizers = {'adam':torch.optim.Adam, 'sgd':torch.optim.SGD}
		loss_functions = {'bce':nn.BCELoss(reduction=reduction), 'ce':nn.CrossEntropyLoss(reduction=reduction), 'mse':nn.MSELoss(reduction=reduction), 'nll':nn.NLLLoss(reduction=reduction), 'dice':dice_loss, 'focal':FocalLoss(num_class=2)}
		if 'name' not in list(optimizer_opts.keys()):
			optimizer_opts['name']='adam'
		self.optimizer = optimizers[optimizer_opts.pop('name')](self.model.parameters(),**optimizer_opts)
		self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O3')
		self.scheduler = Scheduler(optimizer=self.optimizer,opts=scheduler_opts)
		self.n_epoch = n_epoch
		self.validation_dataloader = validation_dataloader
		self.loss_fn = loss_functions[loss_fn]
		self.loss_fn_name = loss_fn
		self.original_loss_fn = copy.deepcopy(loss_functions[loss_fn])
		self.num_train_batches = num_train_batches

	def calc_loss(self, y_pred, y_true):
		return self.loss_fn(y_pred, y_true)

	def calc_val_loss(self, y_pred, y_true):
		return self.original_loss_fn(y_pred, y_true)

	def reset_loss_fn(self):
		self.loss_fn = self.original_loss_fn

	def add_class_balance_loss(self, dataset):
		self.class_weights = dataset.get_class_weights()
		self.original_loss_fn = copy.deepcopy(self.loss_fn)
		self.loss_fn = lambda y_pred,y_true: sum([self.original_loss_fn(y_pred[y_true==i],y_true[y_true==i]) for i in range(2) if sum(y_true==i)])

	def calc_best_confusion(self, y_pred, y_true):
		fpr, tpr, thresholds = roc_curve(y_true, y_pred)
		threshold=thresholds[np.argmin(np.sum((np.array([0,1])-np.vstack((fpr, tpr)).T)**2,axis=1)**.5)]
		y_pred = (y_pred>threshold).astype(int)
		return threshold, pd.DataFrame(confusion_matrix(y_true,y_pred),index=['F','T'],columns=['-','+']).iloc[::-1,::-1].T

	def loss_backward(self,loss):
		with amp.scale_loss(loss,self.optimizer) as scaled_loss:
			scaled_loss.backward()

	#@pysnooper.snoop('train_loop.log')
	def train_loop(self, epoch, train_dataloader):
		self.model.train(True)
		running_loss = 0.
		n_batch = len(train_dataloader.dataset)//train_dataloader.batch_size if self.num_train_batches == None else self.num_train_batches
		for i, batch in enumerate(train_dataloader):
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
				if val_dataloader.dataset.segmentation and self.loss_fn_name!='dice':
					y_true=y_true.squeeze(1)
				if torch.cuda.is_available():
					X = X.cuda()
					y_true=y_true.cuda()
				y_pred = self.model(X)
				if save_predictions:
					if val_dataloader.dataset.segmentation:
						Y['true'].append(torch.flatten(y_true).detach().cpu().numpy().astype(int).flatten())
						Y['pred'].append((torch.argmax(y_pred,dim=1).detach().cpu().numpy()).astype(int).flatten())
					else:
						Y['true'].append(y_true.detach().cpu().numpy().astype(int).flatten())
						Y['pred'].append((y_pred.detach().cpu().numpy()).astype(float).flatten())
				loss = self.calc_val_loss(y_pred,y_true)
				val_loss=loss.item()
				running_loss += val_loss
				print("Epoch {}[{}/{}] Val Loss:{}".format(epoch,i,n_batch,val_loss))
		if print_val_confusion and save_predictions:
			y_pred,y_true = np.hstack(Y['pred']),np.hstack(Y['true'])
			if not val_dataloader.dataset.segmentation and self.loss_fn_name in ['bce','mse']:
				threshold, best_confusion = self.calc_best_confusion(y_pred,y_true)
				print("Epoch {} Val Confusion, Threshold {}:".format(epoch,threshold))
				print(best_confusion)
				y_true = y_true.astype(int)
				y_pred = (y_pred>=threshold).astype(int)
			print(classification_report(y_true,y_pred))
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
				if test_dataloader.dataset.segmentation:
					y_pred.append((torch.argmax(self.model(X),dim=1).detach().cpu().numpy()).astype(int))
				else:
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
