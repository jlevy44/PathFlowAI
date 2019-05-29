import torch, os, numpy as np, pandas as pd
from utils import *
#from large_data_utils import *
from datasets import *
from models import *
from schedulers import *
from visualize import *
import copy
from sampler import ImbalancedDatasetSampler
import argparse
import sqlite3
#from nonechucks import SafeDataLoader as DataLoader
from torch.utils.data import DataLoader
import click

CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def train():
	pass

def train_model_(training_opts):

	dataset_df = pd.read_csv(training_opts['dataset_df']) if os.path.exists(training_opts['dataset_df']) else create_train_val_test(training_opts['train_val_test_splits'],training_opts['patch_info_file'],training_opts['patch_size'])

	norm_dict = get_normalizer(training_opts['normalization_file'], dataset_df, training_opts['patch_info_file'], training_opts['input_dir'], training_opts['target_names'], training_opts['pos_annotation_class'], training_opts['segmentation'], training_opts['patch_size'], training_opts['fix_names'], training_opts['other_annotations'])

	transformers = get_data_transforms(patch_size = training_opts['patch_resize'], mean=norm_dict['mean'], std=norm_dict['std'], resize=True)

	datasets= {set: DynamicImageDataset(dataset_df, set, training_opts['patch_info_file'], transformers, training_opts['input_dir'], training_opts['target_names'], training_opts['pos_annotation_class'], segmentation=training_opts['segmentation'], patch_size=training_opts['patch_size'], fix_names=training_opts['fix_names'], other_annotations=training_opts['other_annotations']) for set in ['train','val']}
	# nc.SafeDataset(

	dataloaders={set: DataLoader(datasets[set], batch_size=training_opts['batch_size'], shuffle=False if (not training_opts['segmentation']) else (set=='train'), num_workers=10, sampler=ImbalancedDatasetSampler(datasets[set]) if (set=='train' and not training_opts['segmentation']) else None) for set in ['train', 'val']}

	model = generate_model(pretrain=training_opts['pretrain'],architecture=training_opts['architecture'],num_classes=training_opts['num_targets'], add_sigmoid=True, n_hidden=training_opts['n_hidden'])

	if torch.cuda.is_available():
		model.cuda()

	if not training_opts['predict']:

		trainer = ModelTrainer(model=model,
					n_epoch=training_opts['n_epoch'],
					validation_dataloader=dataloaders['val'],
					optimizer_opts=dict(name=training_opts['optimizer'],
										lr=training_opts['lr'],
										weight_decay=training_opts['wd']),
					scheduler_opts=dict(scheduler=training_opts['scheduler_type'],
										lr_scheduler_decay=0.5,
										T_max=training_opts['T_max'],
										eta_min=training_opts['eta_min'],
										T_mult=training_opts['T_mult']),
					loss_fn=training_opts['loss_fn'])

		trainer.fit(dataloaders['train'], verbose=True, print_every=1, plot_training_curves=True, plot_save_file=training_opts['training_curve'], print_val_confusion=training_opts['print_val_confusion'], save_val_predictions=training_opts['save_val_predictions'])

		torch.save(trainer.model.state_dict(),training_opts['save_location'])

	else:

		model_dict = torch.load(training_opts['save_location'])

		model = model.load_state_dict(model_dict)

		trainer = ModelTrainer(model=model)

		y_pred = trainer.predict(dataloaders['val'])

		patch_info = dataloaders['val'].dataset.patch_info
		patch_info['y_pred']=y_pred

		conn = sqlite3.connect(training_opts['prediction_save_path'])
		patch_info.to_sql(str(patch_size),con=conn, if_exists='replace')
		conn.close()

@train.command()
@click.option('-s', '--segmentation', is_flag=True, help='Segmentation task.', show_default=True)
@click.option('-p', '--prediction', is_flag=True, help='Predict on model.', show_default=True)
@click.option('-pa', '--pos_annotation_class', default='', help='Annotation Class from which to apply positive labels.', type=click.Path(exists=False), show_default=True)
@click.option('-oa', '--other_annotations', default=[], multiple=True, help='Annotations in image.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--save_location', default='', help='Model Save Location, append with pickle .pkl.', type=click.Path(exists=False), show_default=True)
@click.option('-i', '--input_dir', default='', help='Input directory containing slides and everything.', type=click.Path(exists=False), show_default=True)
@click.option('-ps', '--patch_size', default=224, help='Patch size.',  show_default=True)
@click.option('-pr', '--patch_resize', default=224, help='Patch resized.',  show_default=True)
@click.option('-tg', '--target_names', default=[], multiple=True, help='Targets.', type=click.Path(exists=False), show_default=True)
@click.option('-df', '--dataset_df', default='', help='CSV file with train/val/test and target info.', type=click.Path(exists=False), show_default=True)
@click.option('-fn', '--fix_names', is_flag=True, help='Whether to fix names in dataset_df.', show_default=True)
@click.option('-a', '--architecture', default='alexnet', help='Neural Network Architecture.', type=click.Choice(['alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
											'inception_v3', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'vgg11', 'vgg11_bn',
											'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'deeplabv3_resnet101','deeplabv3_resnet50','fcn_resnet101', 'fcn_resnet50']), show_default=True)
def train_model(segmentation,prediction,pos_annotation_class,other_annotations,save_location,input_dir,patch_size,patch_resize,target_names,dataset_df,fix_names, architecture):
	# add separate pretrain ability on separating cell types, then transfer learn
	command_opts = dict(segmentation=segmentation,
						prediction=prediction,
						pos_annotation_class=pos_annotation_class,
						other_annotations=other_annotations,
						save_location=save_location,
						input_dir=input_dir,
						patch_size=patch_size,
						target_names=target_names,
						dataset_df=dataset_df,
						fix_names=fix_names,
						architecture=architecture,
						patch_resize=patch_resize)

	training_opts = dict(lr=1e-3,
						 wd=1e-3,
						 scheduler_type='warm_restarts',
						 T_max=10,
						 T_mult=2,
						 eta_min=5e-8,
						 optimizer='adam',
						 n_epoch=300,
						 n_hidden=100,
						 pretrain=True,
						 architecture='alexnet',
						 num_targets=1,
						 batch_size=128,
						 normalization_file="normalization_parameters.pkl",
						 training_curve='training_curve.png',
						 dataset_df='dataset.csv',
						 patch_info_file='patch_info.db',
						 input_dir='./input/',
						 target_names='',
						 pos_annotation_class='',
						 other_annotations=[],
						 segmentation=False,
						 loss_fn='bce',
						 save_location='model.pkl',
						 patch_size=512,
						 fix_names=True,
						 print_val_confusion=True,
						 save_val_predictions=True,
						 predict=prediction,
						 prediction_save_path = 'predictions.db',
						 train_val_test_splits=None
						 )
	segmentation_training_opts = copy.deepcopy(training_opts)
	segmentation_training_opts.update(dict(segmentation=True,
											pos_annotation_class='',
											other_annotations=[],
											loss_fn='ce',
											target_names='',
											dataset_df='',
											normalization_file='normalization_segmentation.pkl',
											input_dir='./input/',
											save_location='segmentation_model.pkl',
											patch_size=512,
											fix_names=False,
											save_val_predictions=False,
											train_val_test_splits='train_val_test.pkl'
											))
	if segmentation:
		training_opts = segmentation_training_opts
	for k in command_opts:
		training_opts[k] = command_opts[k]

	train_model_(training_opts)

if __name__=='__main__':

	train()
