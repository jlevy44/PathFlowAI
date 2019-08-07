import torch, os, numpy as np, pandas as pd
from pathflowai.utils import *
#from large_data_utils import *
from pathflowai.datasets import *
from pathflowai.models import *
from pathflowai.schedulers import *
from pathflowai.visualize import *
import copy
from pathflowai.sampler import ImbalancedDatasetSampler
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
	"""Function to train, predict on model.

	Parameters
	----------
	training_opts : dict
		Training options populated from command line.

	"""

	dataset_df = pd.read_csv(training_opts['dataset_df']) if os.path.exists(training_opts['dataset_df']) else create_train_val_test(training_opts['train_val_test_splits'],training_opts['patch_info_file'],training_opts['patch_size'])

	dataset_opts=dict(dataset_df=dataset_df, set='pass', patch_info_file=training_opts['patch_info_file'], input_dir=training_opts['input_dir'], target_names=training_opts['target_names'], pos_annotation_class=training_opts['pos_annotation_class'], segmentation=training_opts['segmentation'], patch_size=training_opts['patch_size'], fix_names=training_opts['fix_names'], other_annotations=training_opts['other_annotations'], target_segmentation_class=training_opts['target_segmentation_class'][0] if set=='train' else -1, target_threshold=training_opts['target_threshold'][0], oversampling_factor=training_opts['oversampling_factor'][0] if set=='train' else 1, n_segmentation_classes=training_opts['num_targets'],gdl=training_opts['loss_fn']=='gdl',mt_bce=training_opts['mt_bce'], classify_annotations=training_opts['classify_annotations'])

	norm_dict = get_normalizer(training_opts['normalization_file'], dataset_opts)

	transform_opts=dict(patch_size = training_opts['patch_resize'], mean=norm_dict['mean'], std=norm_dict['std'], resize=True, transform_platform=training_opts['transform_platform'] if not training_opts['segmentation'] else 'albumentations')

	transformers = get_data_transforms(**transform_opts)

	datasets= {set: DynamicImageDataset(dataset_df, set, training_opts['patch_info_file'], transformers, training_opts['input_dir'], training_opts['target_names'], training_opts['pos_annotation_class'], segmentation=training_opts['segmentation'], patch_size=training_opts['patch_size'], fix_names=training_opts['fix_names'], other_annotations=training_opts['other_annotations'], target_segmentation_class=training_opts['target_segmentation_class'][0] if set=='train' else -1, target_threshold=training_opts['target_threshold'][0], oversampling_factor=training_opts['oversampling_factor'][0] if set=='train' else 1, n_segmentation_classes=training_opts['num_targets'],gdl=training_opts['loss_fn']=='gdl',mt_bce=training_opts['mt_bce'], classify_annotations=training_opts['classify_annotations']) for set in ['train','val','test']}
	# nc.SafeDataset(
	print(datasets['train'])

	if len(training_opts['target_segmentation_class']) > 1:
		from functools import reduce
		for i in range(1,len(training_opts['target_segmentation_class'])):
			#print(training_opts['classify_annotations'])
			datasets['train'].concat(DynamicImageDataset(dataset_df, 'train', training_opts['patch_info_file'], transformers, training_opts['input_dir'], training_opts['target_names'], training_opts['pos_annotation_class'], segmentation=training_opts['segmentation'], patch_size=training_opts['patch_size'], fix_names=training_opts['fix_names'], other_annotations=training_opts['other_annotations'], target_segmentation_class=training_opts['target_segmentation_class'][i], target_threshold=training_opts['target_threshold'][i], oversampling_factor=training_opts['oversampling_factor'][i],n_segmentation_classes=training_opts['num_targets'],gdl=training_opts['loss_fn']=='gdl',mt_bce=training_opts['mt_bce'],classify_annotations=training_opts['classify_annotations']))
		#datasets['train']=reduce(lambda x,y: x.concat(y),[DynamicImageDataset(dataset_df, 'train', training_opts['patch_info_file'], transformers, training_opts['input_dir'], training_opts['target_names'], training_opts['pos_annotation_class'], segmentation=training_opts['segmentation'], patch_size=training_opts['patch_size'], fix_names=training_opts['fix_names'], other_annotations=training_opts['other_annotations'], target_segmentation_class=training_opts['target_segmentation_class'][i], target_threshold=training_opts['target_threshold'][i], oversampling_factor=training_opts['oversampling_factor'][i]) for i in range(len(training_opts['target_segmentation_class']))])
		print(datasets['train'])

	if training_opts['supplement']:
		old_train_set = copy.deepcopy(datasets['train'])
		datasets['train']=DynamicImageDataset(dataset_df, 'train', training_opts['patch_info_file'], transformers, training_opts['input_dir'], training_opts['target_names'], training_opts['pos_annotation_class'], segmentation=training_opts['segmentation'], patch_size=training_opts['patch_size'], fix_names=training_opts['fix_names'], other_annotations=training_opts['other_annotations'], target_segmentation_class=-1, target_threshold=training_opts['target_threshold'], oversampling_factor=1,n_segmentation_classes=training_opts['num_targets'],gdl=training_opts['loss_fn']=='gdl',mt_bce=training_opts['mt_bce'],classify_annotations=training_opts['classify_annotations'])
		datasets['train'].concat(old_train_set)

	if training_opts['subsample_p']<1.0:
		datasets['train'].subsample(training_opts['subsample_p'])

	if training_opts['subsample_p_val']<1.0:
		if training_opts['subsample_p_val']==-1.:
			training_opts['subsample_p_val']=training_opts['subsample_p']
		if training_opts['subsample_p_val']<1.0:
			datasets['val'].subsample(training_opts['subsample_p_val'])

	if training_opts['num_training_images_epoch']>0:
		num_train_batches = min(training_opts['num_training_images_epoch'],len(datasets['train']))//training_opts['batch_size']
	else:
		num_train_batches = None

	if training_opts['classify_annotations']:
		binarizer=datasets['train'].binarize_annotations(num_targets=training_opts['num_targets'],binary_threshold=training_opts['binary_threshold'])
		datasets['val'].binarize_annotations(num_targets=training_opts['num_targets'],binary_threshold=training_opts['binary_threshold'])
		datasets['test'].binarize_annotations(num_targets=training_opts['num_targets'],binary_threshold=training_opts['binary_threshold'])
		training_opts['num_targets']=len(datasets['train'].targets)
	for Set in ['train','val','test']:
		print(datasets[Set].patch_info.iloc[:,6:].sum(axis=0))

	if training_opts['external_test_db'] and training_opts['external_test_dir']:
		datasets['test'].update_dataset(input_dir=training_opts['external_test_dir'],new_db=training_opts['external_test_db'])

	dataloaders={set: DataLoader(datasets[set], batch_size=training_opts['batch_size'], shuffle=False if (not training_opts['segmentation']) else (set=='train'), num_workers=10, sampler=ImbalancedDatasetSampler(datasets[set]) if (training_opts['imbalanced_correction'] and set=='train' and not training_opts['segmentation']) else None) for set in ['train', 'val', 'test']}

	#print(dataloaders) # FIXME VAL SEEMS TO BE MISSING DURING PREDICTION

	model = generate_model(pretrain=training_opts['pretrain'],architecture=training_opts['architecture'],num_classes=training_opts['num_targets'], add_sigmoid=False, n_hidden=training_opts['n_hidden'], segmentation=training_opts['segmentation'])

	if os.path.exists(training_opts['pretrained_save_location']):
		model_dict = torch.load(training_opts['pretrained_save_location'])
		keys=list(model_dict.keys())
		if not training_opts['segmentation']:
			model_dict.update(dict(list(model.state_dict().items())[-2:]))#={k:model_dict[k] for k in keys[:-2]}
		model.load_state_dict(model_dict) # this will likely break after pretraining?

	if torch.cuda.is_available():
		model.cuda()

	if 0 and training_opts['run_test']:
		for X,y in dataloaders['train']:
			np.save('test_predictions.npy',model(X.cuda() if torch.cuda.is_available() else X).detach().cpu().numpy())
			exit()

	model_trainer_opts=dict(model=model,
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
				loss_fn=training_opts['loss_fn'],
				num_train_batches=num_train_batches)

	if not training_opts['predict']:

		trainer = ModelTrainer(**model_trainer_opts)

		if training_opts['imbalanced_correction2']:
			trainer.add_class_balance_loss(datasets['train'])
			if training_opts['adopt_training_loss']:
				trainer.val_loss_fn = trainer.loss_fn

		trainer.fit(dataloaders['train'], verbose=True, print_every=1, plot_training_curves=True, plot_save_file=training_opts['training_curve'], print_val_confusion=training_opts['print_val_confusion'], save_val_predictions=training_opts['save_val_predictions'])

		torch.save(trainer.model.state_dict(),training_opts['save_location'])

	else:

		model_dict = torch.load(training_opts['save_location'])

		model.load_state_dict(model_dict)

		if training_opts['extract_model']:
			dataset_opts.update(dict(target_segmentation_class=-1, target_threshold=training_opts['target_threshold'][0] if len(training_opts['target_threshold']) else 0., set='test', binary_threshold=training_opts['binary_threshold'], num_targets=training_opts['num_targets'], oversampling_factor=1))
			torch.save(dict(model=model,dataset_opts=dataset_opts, transform_opts=transform_opts),'{}.{}'.format(training_opts['save_location'],'extracted_model.pkl'))
			exit()

		trainer = ModelTrainer(**model_trainer_opts)

		if training_opts['segmentation']:
			for ID, dataset in datasets['test'].split_by_ID():
				dataloader = DataLoader(dataset, batch_size=training_opts['batch_size'], shuffle=False, num_workers=10)
				if training_opts['run_test']:
					for X,y in dataloader:
						np.save('test_predictions.npy',model(X.cuda() if torch.cuda.is_available() else X).detach().cpu().numpy())
						exit()
				y_pred = trainer.predict(dataloader)
				print(ID,y_pred.shape)
				segmentation_predictions2npy(y_pred, dataset.patch_info, dataset.segmentation_maps[ID], npy_output='{}/{}_predict.npy'.format(training_opts['prediction_output_dir'],ID), original_patch_size=training_opts['patch_size'], resized_patch_size=training_opts['patch_resize'])
		else:
			extract_embedding=training_opts['extract_embedding']
			if extract_embedding:
				trainer.model.fc = trainer.model.fc[0]
				trainer.bce=False
			y_pred = trainer.predict(dataloaders['test'])

			patch_info = dataloaders['test'].dataset.patch_info

			if extract_embedding:
				patch_info['name']=patch_info.astype(str).apply(lambda x: '\n'.join(['{}:{}'.format(k,v) for k,v in x.to_dict().items()]),axis=1)#.apply(','.join,axis=1)
				embeddings=pd.DataFrame(y_pred,index=patch_info['name'])
				embeddings['ID']=patch_info['ID'].values
				torch.save(dict(embeddings=embeddings,patch_info=patch_info),join(training_opts['prediction_output_dir'],'embeddings.pkl'))

			else:
				if len(y_pred.shape)>1 and y_pred.shape[1]>1:
					annotations = np.vectorize(lambda x: x+'_pred')(np.arange(y_pred.shape[1]).astype(str)).tolist() # [training_opts['pos_annotation_class']]+training_opts['other_annotations']] if training_opts['classify_annotations'] else
					for i in range(y_pred.shape[1]):
						patch_info.loc[:,annotations[i]]=y_pred[:,i]
				patch_info['y_pred']=y_pred if (training_opts['num_targets']==1 or not (training_opts['classify_annotations'] or training_opts['mt_bce'])) else y_pred.argmax(axis=1)

				conn = sqlite3.connect(training_opts['prediction_save_path'])
				patch_info.to_sql(str(training_opts['patch_size']),con=conn, if_exists='replace')
				conn.close()

@train.command()
@click.option('-s', '--segmentation', is_flag=True, help='Segmentation task.', show_default=True)
@click.option('-p', '--prediction', is_flag=True, help='Predict on model.', show_default=True)
@click.option('-pa', '--pos_annotation_class', default='', help='Annotation Class from which to apply positive labels.', type=click.Path(exists=False), show_default=True)
@click.option('-oa', '--other_annotations', default=[], multiple=True, help='Annotations in image.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--save_location', default='', help='Model Save Location, append with pickle .pkl.', type=click.Path(exists=False), show_default=True)
@click.option('-pt', '--pretrained_save_location', default='', help='Model Save Location, append with pickle .pkl, pretrained by previous analysis to be finetuned.', type=click.Path(exists=False), show_default=True)
@click.option('-i', '--input_dir', default='', help='Input directory containing slides and everything.', type=click.Path(exists=False), show_default=True)
@click.option('-ps', '--patch_size', default=224, help='Patch size.',  show_default=True)
@click.option('-pr', '--patch_resize', default=224, help='Patch resized.',  show_default=True)
@click.option('-tg', '--target_names', default=[], multiple=True, help='Targets.', type=click.Path(exists=False), show_default=True)
@click.option('-df', '--dataset_df', default='', help='CSV file with train/val/test and target info.', type=click.Path(exists=False), show_default=True)
@click.option('-fn', '--fix_names', is_flag=True, help='Whether to fix names in dataset_df.', show_default=True)
@click.option('-a', '--architecture', default='alexnet', help='Neural Network Architecture.', type=click.Choice(['alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
											'inception_v3', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'vgg11', 'vgg11_bn','unet','unet2','nested_unet','fast_scnn',
											'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'deeplabv3_resnet101','deeplabv3_resnet50','fcn_resnet101', 'fcn_resnet50']+['efficientnet-b{}'.format(i) for i in range(8)]), show_default=True)
@click.option('-imb', '--imbalanced_correction', is_flag=True, help='Attempt to correct for imbalanced data.', show_default=True)
@click.option('-imb2', '--imbalanced_correction2', is_flag=True, help='Attempt to correct for imbalanced data.', show_default=True)
@click.option('-ca', '--classify_annotations', is_flag=True, help='Classify annotations.', show_default=True)
@click.option('-nt', '--num_targets', default=1, help='Number of targets.', show_default=True)
@click.option('-ss', '--subsample_p', default=1.0, help='Subsample training set.', show_default=True)
@click.option('-ssv', '--subsample_p_val', default=-1., help='Subsample val set. If not set, defaults to that of training set', show_default=True)
@click.option('-t', '--num_training_images_epoch', default=-1, help='Number of training images per epoch. -1 means use all training images each epoch.', show_default=True)
@click.option('-lr', '--learning_rate', default=1e-2, help='Learning rate.', show_default=True)
@click.option('-tp', '--transform_platform', default='torch', help='Transform platform for nonsegmentation tasks.', type=click.Choice(['torch','albumentations']))
@click.option('-ne', '--n_epoch', default=10, help='Number of epochs.', show_default=True)
@click.option('-pi', '--patch_info_file', default='patch_info.db', help='Patch info file.', type=click.Path(exists=False), show_default=True)
@click.option('-tc', '--target_segmentation_class', default=[-1], multiple=True, help='Segmentation Class to finetune on.',  show_default=True)
@click.option('-tt', '--target_threshold', default=[0.], multiple=True, help='Threshold to include target for segmentation if saving one class.',  show_default=True)
@click.option('-ov', '--oversampling_factor', default=[1.], multiple=True, help='How much to oversample training set.',  show_default=True)
@click.option('-sup', '--supplement', is_flag=True, help='Use the thresholding to supplement the original training set.', show_default=True)
@click.option('-bs', '--batch_size', default=10, help='Batch size.',  show_default=True)
@click.option('-rt', '--run_test', is_flag=True, help='Output predictions for a batch to "test_predictions.npy". Use for debugging.',  show_default=True)
@click.option('-mtb', '--mt_bce', is_flag=True, help='Run multi-target bce predictions on the annotations.',  show_default=True)
@click.option('-po', '--prediction_output_dir', default='predictions', help='Where to output segmentation predictions.', type=click.Path(exists=False), show_default=True)
@click.option('-ee', '--extract_embedding', is_flag=True, help='Extract embeddings.',  show_default=True)
@click.option('-em', '--extract_model', is_flag=True, help='Save entire torch model.',  show_default=True)
@click.option('-bt', '--binary_threshold', default=0., help='If running binary classification on annotations, dichotomize selected annotation as such.',  show_default=True)
@click.option('-prt', '--pretrain', is_flag=True, help='Pretrain on ImageNet.', show_default=True)
@click.option('-olf', '--overwrite_loss_fn', default='', help='Overwrite the default training loss functions with loss of choice.', type=click.Choice(['','bce','mse','focal','dice','gdl','ce']), show_default=True)
@click.option('-atl', '--adopt_training_loss', is_flag=True, help='Adopt training loss function for validation calculation.', show_default=True)
@click.option('-tdb', '--external_test_db', default='', help='External database of samples to test on.', type=click.Path(exists=False), show_default=True)
@click.option('-tdir', '--external_test_dir', default='', help='External directory of samples to test on.', type=click.Path(exists=False), show_default=True)
def train_model(segmentation,prediction,pos_annotation_class,other_annotations,save_location,pretrained_save_location,input_dir,patch_size,patch_resize,target_names,dataset_df,fix_names, architecture, imbalanced_correction, imbalanced_correction2, classify_annotations, num_targets, subsample_p,subsample_p_val,num_training_images_epoch, learning_rate, transform_platform, n_epoch, patch_info_file, target_segmentation_class, target_threshold, oversampling_factor, supplement, batch_size, run_test, mt_bce, prediction_output_dir, extract_embedding, extract_model, binary_threshold, pretrain, overwrite_loss_fn, adopt_training_loss, external_test_db,external_test_dir):
	"""Train and predict using model for regression and classification tasks."""
	# add separate pretrain ability on separating cell types, then transfer learn
	# add pretrain and efficient net, pretraining remove last layer while loading state dict
	target_segmentation_class=list(map(int,target_segmentation_class))
	target_threshold=list(map(float,target_threshold))
	oversampling_factor=[(int(x) if float(x)>=1 else float(x)) for x in oversampling_factor]
	other_annotations=list(other_annotations)
	command_opts = dict(segmentation=segmentation,
						prediction=prediction,
						pos_annotation_class=pos_annotation_class,
						other_annotations=other_annotations,
						save_location=save_location,
						pretrained_save_location=pretrained_save_location,
						input_dir=input_dir,
						patch_size=patch_size,
						target_names=target_names,
						dataset_df=dataset_df,
						fix_names=fix_names,
						architecture=architecture,
						patch_resize=patch_resize,
						imbalanced_correction=imbalanced_correction,
						imbalanced_correction2=imbalanced_correction2,
						classify_annotations=classify_annotations,
						num_targets=num_targets,
						subsample_p=subsample_p,
						num_training_images_epoch=num_training_images_epoch,
						lr=learning_rate,
						transform_platform=transform_platform,
						n_epoch=n_epoch,
						patch_info_file=patch_info_file,
						target_segmentation_class=target_segmentation_class,
						target_threshold=target_threshold,
						oversampling_factor=oversampling_factor,
						supplement=supplement,
						predict=prediction,
						batch_size=batch_size,
						run_test=run_test,
						mt_bce=mt_bce,
						prediction_output_dir=prediction_output_dir,
						extract_embedding=extract_embedding,
						extract_model=extract_model,
						binary_threshold=binary_threshold,
						subsample_p_val=subsample_p_val,
						wd=1e-3,
						scheduler_type='warm_restarts',
						T_max=10,
						T_mult=2,
						eta_min=5e-8,
						optimizer='adam',
						n_hidden=100,
						pretrain=pretrain,
						training_curve='training_curve.png',
						adopt_training_loss=adopt_training_loss,
						external_test_db=external_test_db,
						external_test_dir=external_test_dir)

	training_opts = dict(normalization_file="normalization_parameters.pkl",
						 loss_fn='bce',
						 print_val_confusion=True,
						 save_val_predictions=True,
						 prediction_save_path = 'predictions.db',
						 train_val_test_splits='train_val_test.pkl'
						 )
	segmentation_training_opts = copy.deepcopy(training_opts)
	segmentation_training_opts.update(dict(loss_fn='dice',#gdl dice+ce
											normalization_file='normalization_segmentation.pkl',
											fix_names=False,
											save_val_predictions=True,
											))
	if segmentation:
		training_opts = segmentation_training_opts
	for k in command_opts:
		training_opts[k] = command_opts[k]
	if classify_annotations:
		if training_opts['num_targets']==1:
			training_opts['loss_fn']='bce'
		else:
			training_opts['loss_fn']='ce'
	if mt_bce:
		training_opts['loss_fn']='bce'
	if overwrite_loss_fn:
		training_opts['loss_fn']=overwrite_loss_fn

	train_model_(training_opts)


if __name__=='__main__':

	train()
