import argparse
import os
from os.path import join
from utils import run_preprocessing_pipeline, generate_patch_pipeline, img2npy_
import click


CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def preprocessing():
    pass

def output_if_exists(filename):
    if os.path.exists(filename):
        return filename
    return None

@preprocessing.command()
@click.option('-npy', '--img2npy', is_flag=True, help='Image to numpy for faster read.', show_default=True)
@click.option('-b', '--basename', default='A01', help='Basename of patches.', type=click.Path(exists=False), show_default=True)
@click.option('-i', '--input_dir', default='./inputs/', help='Input directory for patches.', type=click.Path(exists=False), show_default=True)
@click.option('-a', '--annotations', default=[], multiple=True, help='Annotations in image in order.', type=click.Path(exists=False), show_default=True)
@click.option('-pr', '--preprocess', is_flag=True, help='Run preprocessing pipeline.', show_default=True)
@click.option('-pa', '--patches', is_flag=True, help='Add patches to SQL.', show_default=True)
@click.option('-t', '--threshold', default=0.05, help='Threshold to remove non-purple slides.',  show_default=True)
@click.option('-ps', '--patch_size', default=224, help='Patch size.',  show_default=True)
def preprocess_pipeline(img2npy,basename,input_dir,annotations,preprocess,patches,threshold,patch_size):

    for ext in ['.npy','.svs','.tiff','.tif']:
        svs_file = output_if_exists(join(input_dir,'{}{}'.format(basename,ext)))
        if svs_file != None:
            break

    if img2npy and not svs_file.endswith('.npy'):
        svs_file = img2npy_(input_dir,basename, svs_file)

    xml_file = output_if_exists(join(input_dir,'{}.xml'.format(basename)))
    npy_mask = output_if_exists(join(input_dir,'{}_mask.npy'.format(basename)))
    out_zarr = join(input_dir,'{}.zarr'.format(basename))
    out_pkl = join(input_dir,'{}_mask.pkl'.format(basename))
    out_db = join('.','patch_info.db'.format(basename))

    if run_preprocess:
        run_preprocessing_pipeline(svs_file=svs_file,
                               xml_file=xml_file,
                               npy_mask=npy_mask,
                               annotations=annotations,
                               out_zarr=out_zarr,
                               out_pkl=out_pkl)

    if run_patches: # ADD EXPORT TO SQL, TABLE NAME IS PATCH SIZE
        generate_patch_pipeline(basename,
                            input_dir=input_dir,
                            annotations=annotations,
                            threshold=threshold,
                            patch_size=patch_size,
                            out_db=out_db)

if __name__ == '__main__':
    preprocessing()
