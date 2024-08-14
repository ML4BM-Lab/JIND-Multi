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

# run-main --PATH path/to/data --BATCH_COL batch_column_name --LABELS_COL labels_column_name --SOURCE_DATASET_NAME source_name --TARGET_DATASET_NAME target_name --OUTPUT_PATH output_path --NUM_FEATURES 5000 --MIN_CELL_TYPE_POPULATION 100 --N_TRIAL 1 --USE_GPU True
# compare-methods --PATH path/to/data --BATCH_COL batch_column_name --LABELS_COL labels_column_name --SOURCE_DATASET_NAME source_name --TARGET_DATASET_NAME target_name --OUTPUT_PATH output_path --NUM_FEATURES 5000 --MIN_CELL_TYPE_POPULATION 100 --N_TRIAL 1 --USE_GPU True

# jind_multi.main:main indica que el comando jind-multi ejecutará la función main del módulo main en el paquete jind_multi.

# Ejemplo de Uso
# Supongamos que tienes una función main en tu módulo jind_multi/main.py que es el punto de entrada principal de tu aplicación. Con la configuración de setup.py proporcionada:

# Instalas el paquete localmente:

# bash
# Copiar código
# pip install .
# Ejecutas el comando jind-multi desde la terminal:

# bash
# Copiar código
# jind-multi --arg1 value1 --arg2 value2
# Esto invocará la función main en jind_multi/main.py y pasará los argumentos de la línea de comandos a esa función.

# Ventajas de Usar Entry Points
# Facilidad de Uso: Permite que tu aplicación sea ejecutada fácilmente desde la línea de comandos sin tener que invocar directamente el script Python.

# Interoperabilidad: Puedes definir múltiples entry points para diferentes funcionalidades dentro de tu paquete.

# Despliegue Simplificado: Los usuarios pueden instalar tu paquete y ejecutar comandos específicos sin preocuparse por la estructura interna del código.