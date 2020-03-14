# -*- coding: utf-8 -*-
from setuptools import setup

scripts = ["bin/install_apex", "bin/install_lightnet"]

packages = ["pathflowai"]

package_data = {"": ["*"]}

install_requires = [
    "GPUtil>=1.4.0,<2.0.0",
    "Shapely>=1.7.0,<2.0.0",
    "albumentations>=0.4.5,<0.5.0",
    "beautifulsoup4>=4.8.2,<5.0.0",
    "blosc>=1.8.3,<2.0.0",
    "brambox>=3.0.0,<4.0.0",
    "click>=7.1.1,<8.0.0",
    "dask-image>=0.2.0,<0.3.0",
    "dask[dataframe]>=2.12.0,<3.0.0",
    "distributed>=2.12.0,<3.0.0",
    "h5py>=2.10.0,<3.0.0",
    "matplotlib>=3.2.0,<4.0.0",
    "networkx>=2.4,<3.0",
    "nonechucks>=0.4.0,<0.5.0",
    "numcodecs>=0.6.4,<0.7.0",
    "numpy>=1.18.1,<2.0.0",
    "opencv-python>=4.2.0,<5.0.0",
    "openslide-python>=1.1.1,<2.0.0",
    "pandas>=1.0.1,<2.0.0",
    "plotly>=4.5.4,<5.0.0",
    "pysnooper>=0.3.0,<0.4.0",
    "pytorchcv>=0.0.57,<0.0.58",
    "pyyaml>=5.3,<6.0",
    "scikit-image>=0.16.2,<0.17.0",
    "scikit-learn>=0.22.2,<0.23.0",
    "scipy>=1.4.1,<2.0.0",
    "seaborn>=0.10.0,<0.11.0",
    "shap>=0.35.0,<0.36.0",
    "tifffile>=2020.2.16,<2021.0.0",
    "torch-encoding>=1.0.1,<2.0.0",
    "torch>=1.4.0,<2.0.0",
    "torchvision>=0.5.0,<0.6.0",
    "umap-learn>=0.3.10,<0.4.0",
    "xarray>=0.15.0,<0.16.0",
    "zarr>=2.4.0,<3.0.0",
]

entry_points = {
    "console_scripts": [
        'pathflowai-monitor = "pathflowai.monitor_memory_usage:monitor"',
        'pathflowai-preprocess = "pathflowai.cli_preprocessing:preprocessing"',
        'pathflowai-train_model = "pathflowai.model_training:train"',
        'pathflowai-visualize = "pathflowai.cli_visualizations:visualize"',
    ]
}

setup_kwargs = {
    "name": "pathflowai",
    "version": "0.1.1",
    "description": "A modular approach for preprocessing and deep learning on histopathology images.",
    "long_description": "A Convenient High-Throughput Workflow for Preprocessing, Deep Learning Analytics and Interpretation in Digital Pathology",
    "author": "Joshua Levy",
    "author_email": "joshualevy44@berkeley.edu",
    "maintainer": None,
    "maintainer_email": None,
    "url": "https://github.com/jlevy44/PathFlowAI",
    "license": "MIT",
    "scripts": scripts,
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "entry_points": entry_points,
    "python_requires": ">=3.6.1,<4.0.0",
}


setup(**setup_kwargs)
