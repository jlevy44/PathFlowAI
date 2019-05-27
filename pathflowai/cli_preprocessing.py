import argparse
import os
from os.path import join
from utils import run_preprocessing_pipeline, generate_patch_pipeline

def output_if_exists(filename):
    if os.path.exists(filename):
        return filename
    return None

def main(args):

    basename = args.basename
    input_dir = args.input_dir
    annotations = args.annotations
    run_preprocess = args.preprocess
    run_patches = args.patches
    threshold = args.threshold
    patch_size = args.patch_size

    for ext in ['.npy','.svs','.tiff','.tif']:
        svs_file = output_if_exists(join(input_dir,'{}{}'.format(basename,ext)))
        if svs_file != None:
            break
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
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--preprocess', action='store_true',help='Preprocess')
    parser.add_argument('--patches', action='store_true',help='Patches')
    parser.add_argument('--basename', default='',type=str,help='Basename image')
    parser.add_argument('--input_dir', default='',type=str,help='Directory input')
    parser.add_argument('--annotations', default=[],type=str,nargs='*',help='Annotations in image')
    parser.add_argument('--threshold', default=0.05,type=float,help='How much patch purple until declare purple')
    parser.add_argument('--patch_size', default=224,type=int,help='Size patches.')
    args = parser.parse_args()

    main(args)
