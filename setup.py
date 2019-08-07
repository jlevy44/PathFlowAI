from setuptools import setup
from setuptools.command.install import install
import subprocess
import os
PACKAGES=[  'pandas==0.25.0',
            'numpy',
            'dask[dataframe]',
            'distributed',
            'nonechucks',
            'dask-image',
            'opencv-python',
            'scikit-learn',
            'scipy',
            'umap-learn',
            'pysnooper',
            'tifffile',
            'seaborn',
            'scikit-image',
            'openslide-python',
            'Shapely',
            'click==6.7',
            'torch',
            'torchvision',
            'albumentations',
            'GPUtil',
            'beautifulsoup4',
            'plotly',
            'xarray',
            'matplotlib',
            'networkx',
            'shap']

with open('README.md','r', encoding='utf-8') as f:
      long_description = f.read()

class CustomInstallCommand(install):
    """Custom install setup to help run shell commands (outside shell) before installation"""
    def run(self):
        #for package in PACKAGES:
        #os.system('pip install {}'.format(package))#install.do_egg_install(self)
        self.do_egg_install()#install.run(self)
        subprocess.call('rm -rf apex'.split())
        os.system('git clone https://github.com/NVIDIA/apex')
        #try:
        #os.system('cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./')
        #except:
        os.system('echo pwd && cd apex && (pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ || pip install -v --no-cache-dir ./)')
        subprocess.call('rm -rf apex'.split())

setup(name='pathflowai',
      version='0.1.1',
      description='A modular approach for preprocessing and deep learning on histopathology images.',
      url='https://github.com/jlevy44/PathFlowAI',
      author='Joshua Levy',
      author_email='joshualevy44@berkeley.edu',
      license='MIT',
      scripts=['bin/install_apex'],
      #cmdclass={'install': CustomInstallCommand},
      entry_points={
            'console_scripts':['pathflowai-preprocess=pathflowai.cli_preprocessing:preprocessing',
                               'pathflowai-visualize=pathflowai.cli_visualizations:visualize',
                               'pathflowai-monitor=pathflowai.monitor_memory_usage:monitor',
                               'pathflowai-train_model=pathflowai.model_training:train']
      },
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=['pathflowai'],
      install_requires=PACKAGES)
