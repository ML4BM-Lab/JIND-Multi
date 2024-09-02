import setuptools
from setuptools import setup

setup(
    name='jind_multi',
    version='1.0.0',
    description='JIND-Multi is an extension of the JIND framework designed for the automated annotation of single-cell RNA sequencing (scRNA-Seq) data and scATAC-Seq data',
    url='https://github.com/ML4BM-Lab/JIND-Multi',
    author='Joseba Sancho Zamora',
    packages=setuptools.find_packages(),
    install_requires=[
        "h5py==3.8.0",
        "matplotlib==3.2.1",
        "numpy>=1.19,<1.21",
        "pandas==1.3.5",
        "scanpy==1.8.0", # "scanpy==1.4.6",
        "scikit-learn==0.24.2",
        "scipy==1.5.4",
        "seaborn==0.11.2",
        "sklearn==0.0",
        "statsmodels==0.12.2",
        "torch==1.7.1",
        "torchvision==0.8.2",
        "tornado==6.0.3",
        "tqdm==4.43.0",
        "plotly",
        "kaleido",
        "notebook",
        "ipykernel",
        "openpyxl",
    ],
    extras_require={
        'dev': ['Cython==0.28.5'],
    },

    entry_points={
        'console_scripts': [
            'run-jind-multi=cluster.main:main',
            'compare-methods=cluster.compare_methods:main',
        ],
    },
    zip_safe=False
)
