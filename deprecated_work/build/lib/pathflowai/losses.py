"""
losses.py
=======================
Some additional loss functions that can be called using the pipeline, some of which still to be implemented.
"""

import torch, numpy as np
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union
from torch import Tensor, einsum
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as distance
from torch import nn

def assert_(condition, message='', exception_type=AssertionError):
	"""https://raw.githubusercontent.com/inferno-pytorch/inferno/0561e8a95cde6bfc5e10a3609841b7b0ca5b03ca/inferno/utils/exceptions.py
	Like assert, but with arbitrary exception types."""
	if not condition:
		raise exception_type(message)

class ShapeError(ValueError): # """https://raw.githubusercontent.com/inferno-pytorch/inferno/0561e8a95cde6bfc5e10a3609841b7b0ca5b03ca/inferno/utils/exceptions.py"""
	pass



def flatten_samples(input_):
	"""
	https://raw.githubusercontent.com/inferno-pytorch/inferno/0561e8a95cde6bfc5e10a3609841b7b0ca5b03ca/inferno/utils/torch_utils.py
	Flattens a tensor or a variable such that the channel axis is first and the sample axis
	is second. The shapes are transformed as follows:
		(N, C, H, W) --> (C, N * H * W)
		(N, C, D, H, W) --> (C, N * D * H * W)
		(N, C) --> (C, N)
	The input must be atleast 2d.
	"""
	assert_(input_.dim() >= 2,
			"Tensor or variable must be atleast 2D. Got one of dim {}."
			.format(input_.dim()),
			ShapeError)
	# Get number of channels
	num_channels = input_.size(1)
	# Permute the channel axis to first
	permute_axes = list(range(input_.dim()))
	permute_axes[0], permute_axes[1] = permute_axes[1], permute_axes[0]
	# For input shape (say) NCHW, this should have the shape CNHW
	permuted = input_.permute(*permute_axes).contiguous()
	# Now flatten out all but the first axis and return
	flattened = permuted.view(num_channels, -1)
	return flattened

class GeneralizedDiceLoss(nn.Module):
	"""
	https://raw.githubusercontent.com/inferno-pytorch/inferno/0561e8a95cde6bfc5e10a3609841b7b0ca5b03ca/inferno/extensions/criteria/set_similarity_measures.py
	Computes the scalar Generalized Dice Loss defined in https://arxiv.org/abs/1707.03237

	This version works for multiple classes and expects predictions for every class (e.g. softmax output) and
	one-hot targets for every class.
	"""
	def __init__(self, weight=None, channelwise=False, eps=1e-6, add_softmax=False):
		super(GeneralizedDiceLoss, self).__init__()
		self.register_buffer('weight', weight)
		self.channelwise = channelwise
		self.eps = eps
		self.add_softmax = add_softmax

	def forward(self, input, target):
		"""
		input: torch.FloatTensor or torch.cuda.FloatTensor
		target:     torch.FloatTensor or torch.cuda.FloatTensor

		Expected shape of the inputs:
			- if not channelwise: (batch_size, nb_classes, ...)
			- if channelwise:     (batch_size, nb_channels, nb_classes, ...)
		"""
		assert input.size() == target.size()
		if self.add_softmax:
			input = F.softmax(input, dim=1)
		if not self.channelwise:
			# Flatten input and target to have the shape (nb_classes, N),
			# where N is the number of samples
			input = flatten_samples(input)
			target = flatten_samples(target).float()

			# Find classes weights:
			sum_targets = target.sum(-1)
			class_weigths = 1. / (sum_targets * sum_targets).clamp(min=self.eps)

			# Compute generalized Dice loss:
			numer = ((input * target).sum(-1) * class_weigths).sum()
			denom = ((input + target).sum(-1) * class_weigths).sum()

			loss = 1. - 2. * numer / denom.clamp(min=self.eps)
		else:
			def flatten_and_preserve_channels(tensor):
				tensor_dim = tensor.dim()
				assert tensor_dim >= 3
				num_channels = tensor.size(1)
				num_classes = tensor.size(2)
				# Permute the channel axis to first
				permute_axes = list(range(tensor_dim))
				permute_axes[0], permute_axes[1], permute_axes[2] = permute_axes[1], permute_axes[2], permute_axes[0]
				permuted = tensor.permute(*permute_axes).contiguous()
				flattened = permuted.view(num_channels, num_classes, -1)
				return flattened

			# Flatten input and target to have the shape (nb_channels, nb_classes, N)
			input = flatten_and_preserve_channels(input)
			target = flatten_and_preserve_channels(target)

			# Find classes weights:
			sum_targets = target.sum(-1)
			class_weigths = 1. / (sum_targets * sum_targets).clamp(min=self.eps)

			# Compute generalized Dice loss:
			numer = ((input * target).sum(-1) * class_weigths).sum(-1)
			denom = ((input + target).sum(-1) * class_weigths).sum(-1)

			channelwise_loss = 1. - 2. * numer / denom.clamp(min=self.eps)

			if self.weight is not None:
				if channelwise_loss.dim() == 2:
					channelwise_loss = channelwise_loss.squeeze(1)
				assert self.weight.size() == channelwise_loss.size(),\
					"""`weight` should have shape (nb_channels, ),
					   `target` should have shape (batch_size, nb_channels, nb_classes, ...)"""
				# Apply channel weights:
				channelwise_loss = self.weight * channelwise_loss

			loss = channelwise_loss.sum()

		return loss

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


def uniq(a: Tensor) -> Set:
	"""https://raw.githubusercontent.com/LIVIAETS/surface-loss/master/utils.py"""
	return set(torch.unique(a.cpu()).numpy())

def sset(a: Tensor, sub: Iterable) -> bool:
	"""https://raw.githubusercontent.com/LIVIAETS/surface-loss/master/utils.py"""
	return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
	"""https://raw.githubusercontent.com/LIVIAETS/surface-loss/master/utils.py"""
	return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
	"""https://raw.githubusercontent.com/LIVIAETS/surface-loss/master/utils.py"""
	_sum = t.sum(axis).type(torch.float32)
	_ones = torch.ones_like(_sum, dtype=torch.float32)
	return torch.allclose(_sum, _ones)

def one_hot(t: Tensor, axis=1) -> bool:
	"""https://raw.githubusercontent.com/LIVIAETS/surface-loss/master/utils.py"""
	return simplex(t, axis) and sset(t, [0, 1])

def class2one_hot(seg: Tensor, C: int) -> Tensor:
	"""https://raw.githubusercontent.com/LIVIAETS/surface-loss/master/utils.py"""
	if len(seg.shape) == 2:  # Only w, h, used by the dataloader
		seg = seg.unsqueeze(dim=0)
	assert sset(seg, list(range(C)))

	b, w, h = seg.shape  # type: Tuple[int, int, int]

	res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
	assert res.shape == (b, C, w, h)
	assert one_hot(res)

	return res

def one_hot2dist(seg: np.ndarray) -> np.ndarray:
	"""https://raw.githubusercontent.com/LIVIAETS/surface-loss/master/utils.py"""
	assert one_hot(torch.Tensor(seg), axis=0)
	C: int = len(seg)

	res = np.zeros_like(seg)
	for c in range(C):
		posmask = seg[c].astype(np.bool)

		if posmask.any():
			negmask = ~posmask
			res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
	return res

class SurfaceLoss():
	"""https://raw.githubusercontent.com/LIVIAETS/surface-loss/master/losses.py"""
	def __init__(self, **kwargs):
		# Self.idc is used to filter out some classes of the target mask. Use fancy indexing
		self.idc: List[int] = kwargs["idc"]
		print(f"Initialized {self.__class__.__name__} with {kwargs}")

	def __call__(self, probs: Tensor, dist_maps: Tensor, _: Tensor) -> Tensor:
		assert simplex(probs)
		assert not one_hot(dist_maps)

		pc = probs[:, self.idc, ...].type(torch.float32)
		dc = dist_maps[:, self.idc, ...].type(torch.float32)

		multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

		loss = multipled.mean()

		return loss

class GeneralizedDice():
	"""https://raw.githubusercontent.com/LIVIAETS/surface-loss/master/losses.py"""
	def __init__(self, **kwargs):
		# Self.idc is used to filter out some classes of the target mask. Use fancy indexing
		self.idc: List[int] = kwargs["idc"]
		print(f"Initialized {self.__class__.__name__} with {kwargs}")

	def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
		assert simplex(probs) and simplex(target)

		pc = probs[:, self.idc, ...].type(torch.float32)
		tc = target[:, self.idc, ...].type(torch.float32)

		w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
		intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
		union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

		divided: Tensor = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

		loss = divided.mean()

		return loss
