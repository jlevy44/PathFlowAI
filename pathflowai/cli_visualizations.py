import click
from visualize import PredictionPlotter, plot_svs_image
import glob, os
#from utils import *
import dask.array as da


CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def visualize():
    pass

@visualize.command()
@click.option('-i', '--input_dir', default='./inputs/', help='Input directory for patches.', type=click.Path(exists=False), show_default=True)
@click.option('-b', '--basename', default='A01', help='Basename of patches.', type=click.Path(exists=False), show_default=True)
@click.option('-p', '--patch_info_file', default='patch_info.db', help='Datbase containing all patches', type=click.Path(exists=False), show_default=True)
@click.option('-ps', '--patch_size', default=224, help='Patch size.',  show_default=True)
@click.option('-x', '--x', default=0, help='X Coordinate of patch.',  show_default=True)
@click.option('-y', '--y', default=0, help='Y coordinate of patch.',  show_default=True)
@click.option('-o', '--outputfname', default='./output_image.png', help='Output extracted image.', type=click.Path(exists=False), show_default=True)
def extract_patch(input_dir, basename, patch_info_file, patch_size, x, y, outputfname):
    dask_arr_dict = {os.path.basename(f).split('.zarr')[0]:da.from_zarr(f) for f in glob.glob(os.path.join(input_dir,'*.zarr')) if os.path.basename(f).split('.zarr')[0] == basename}
    pred_plotter = PredictionPlotter(dask_arr_dict, patch_info_file, compression_factor=3, alpha=0.5, patch_size=patch_size, no_db=True)
    img = pred_plotter.return_patch(basename, x, y, patch_size)
    pred_plotter.output_image(img,outputfname)

@visualize.command()
@click.option('-s', '--svs_file', default='./inputs/a.svs', help='Input svs file.', type=click.Path(exists=False), show_default=True)
@click.option('-cf', '--compression_factor', default=3., help='How much compress image.',  show_default=True)
@click.option('-o', '--outputfname', default='./output_image.png', help='Output extracted image.', type=click.Path(exists=False), show_default=True)
def plot_svs(svs_file, compression_factor, outputfname):
    plot_svs_image(svs_file, compression_factor=compression_factor, test_image_name=outputfname)

@visualize.command()
@click.option('-i', '--input_dir', default='./inputs/', help='Input directory for patches.', type=click.Path(exists=False), show_default=True)
@click.option('-b', '--basename', default='A01', help='Basename of patches.', type=click.Path(exists=False), show_default=True)
@click.option('-p', '--patch_info_file', default='patch_info.db', help='Datbase containing all patches', type=click.Path(exists=False), show_default=True)
@click.option('-ps', '--patch_size', default=224, help='Patch size.',  show_default=True)
@click.option('-o', '--outputfname', default='./output_image.png', help='Output extracted image.', type=click.Path(exists=False), show_default=True)
@click.option('-an', '--annotations', is_flag=True, help='Plot annotations instead of predictions.', show_default=True)
@click.option('-cf', '--compression_factor', default=3., help='How much compress image.',  show_default=True)
@click.option('-al', '--alpha', default=0.8, help='How much to give annotations/predictions versus original image.',  show_default=True)
def plot_predictions(input_dir,basename,patch_info_file,patch_size,outputfname,annotations, compression_factor, alpha):
    dask_arr_dict = {os.path.basename(f).split('.zarr')[0]:da.from_zarr(f) for f in glob.glob(os.path.join(input_dir,'*.zarr')) if os.path.basename(f).split('.zarr')[0] == basename}
    pred_plotter = PredictionPlotter(dask_arr_dict, patch_info_file, compression_factor=compression_factor, alpha=alpha, patch_size=patch_size, no_db=False, plot_annotation=annotations)
    img = pred_plotter.generate_image(basename)
    pred_plotter.output_image(img, outputfname)


if __name__ == '__main__':
    visualize()
