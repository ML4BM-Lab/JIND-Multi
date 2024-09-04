# JIND-Multi  
<!-- #### Publication:   -->

<p align="center">
  <a href="https://www.python.org/downloads/release/python-368/">
    <img src="https://img.shields.io/badge/Python-3.6%2B-blue.svg" alt="Python Version">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-lightgrey.svg" alt="Platform">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/GPU-Supported-brightgreen.svg" alt="GPU Support">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </a>
</p>

**JIND-Multi** is an advanced extension of the JIND framework, designed to automate the annotation of single-cell RNA sequencing (scRNA-Seq). This framework, originally introduced by [Goyal et al., 2022](https://academic.oup.com/bioinformatics/article/38/9/2488/6543609), now supports the transfer of cell-type labels from multiple annotated datasets, enhancing the accuracy and reliability of annotations. Additionally, **JIND-Multi** is applicable for annotating scATAC-Seq data and can flag cells as "unassigned" if predictions fall below predefined thresholds.

Leveraging multiple annotated datasets, such as those in an atlas, **JIND-Multi** improves the precision of unlabeled dataset annotations while reducing rejection rates (unassigned cells). We offer a robust and efficient implementation of **JIND-Multi**, available for the scientific community.

<p align="center">
    <img src="https://github.com/ML4BM-Lab/JIND-Multi/blob/master/JIND.png" alt="JIND-Multi Logo" width="700">
</p>

## Prerequisites

- **Operating System:** Linux or macOS
- **Environment Manager:** [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- **Programming Language:** Python 3.6 or higher (tested on 3.6.8 and 3.7.11)
- **Hardware:** CPU or NVIDIA GPU + CUDA CuDNN

## Installation
To install **JIND-Multi**, follow these steps:

```bash
git clone https://github.com/ML4BM-Lab/JIND-Multi.git
cd JIND-Multi
conda create -n jind python=3.7.16 
conda activate jind
pip install -e .
```

## Data
The datasets used to reproduce the results presented in the manuscript are available at the following link: https://doi.org/10.5281/zenodo.11098805.

Please note that if you are using any of the datasets published on Zenodo, refer to the table [`Input Arguments Information`](#input-arguments-information) at the end of this README to correctly add the input arguments.

# Executing JIND-Multi
There are two options to execute the JIND-Multi framework: 
* Running the Python script 
* Submitting a job to a HPC queue
* Running with Docker

### Option 1: The Python Script 
For executing JIND-Multi on the `Brain Neurips` dataset, you can use a `.json` configuration file. The content of the file should be as follows:

```json
{
    "PATH": "/path/to/data/All_human_brain.h5ad",
    "BATCH_COL": "batch",
    "LABELS_COL": "label",
    "SOURCE_DATASET_NAME": "C4",
    "TARGET_DATASET_NAME": "C7",
    "OUTPUT_PATH": "/path/to/save/results",
    "PRETRAINED_MODEL_PATH": "/path/to/pretrained_model_folder",
    "TRAIN_DATASETS_NAMES": "['AD2', 'ADx1']",
    "NUM_FEATURES": 5000,
    "MIN_CELL_TYPE_POPULATION": 100,
    "USE_GPU": true
}
```

where,
- **`PATH`**: (string) Path to the file with the data in a `.h5ad` format.
- **`BATCH_COL`**: (string) Column name with the information of the different batches or donors in your AnnData object.
- **`LABELS_COL`**: (string) Column name with the different cell types in your AnnData object.
- **`SOURCE_DATASET_NAME`**: (string) Optional. name of the source batch. If no batch is specified, JIND-Multi will select as source batch the sample that produces the least amount of rejected cells on the target batch when used as source in JIND (i.e., without additional intermediate batches).
- **`TARGET_DATASET_NAME`**: (string) Name of the target batch to which transfer the annotations from the rest of the datasets.
- **`OUTPUT_PATH`**: (string) Path where the model performance results will be stored. 
- **`PRETRAINED_MODEL_PATH`**: (string) Optional. This argument specifies the path to a folder containing pre-trained models. If this path is provided, JIND-Multi will use the models from this folder instead of training new ones to infer on the new target batch. The folder should contain model files `.pt` format and a `.json` file containing the predictions on the validation test set used to compute the thresholds. If this argument is not provided or left empty, the script will proceed to train a new model from scratch based on the provided data.
- **`TRAIN_DATASETS_NAMES`**: (string) Optional. This setting allows to specify the order of intermediate datasets used for training. The source batch should not be included here. If no specific order is provided, the model will train on the intermediate datasets in the order they appear in the data.
- **`NUM_FEATURES`**: (int) Optional. Number of genes to consider for modeling, default is 5000.
- **`MIN_CELL_TYPE_POPULATION`**: (int) Optional. For each batch, the minimum necessary number of cells per cell type to train. If this requirement is not met in any batch, the cells belonging to this cell type are discarded from all batches, the default is 100 cells.
- **`USE_GPU`**: (bool) Optional but recommended. Whether to use the GPU to train, default is True.

**Note:** The `PRETRAINED_MODEL_PATH` argument is optional and should be provided only if you want to use a pre-trained model. If you do not specify this argument, JIND-Multi will train a new model from scratch based on the provided data.

To execute JIND-Multi using the configuration file:

```bash
run-jind-multi --config /path/to/config.json
```

Alternatively, you can run JIND-Multi directly from the command line by providing all the necessary parameters:

```bash
run-jind-multi --PATH "/path/to/data/All_human_brain.h5ad" \
               --BATCH_COL "batch" \
               --LABELS_COL "label" \
               --SOURCE_DATASET_NAME "C4" \
               --TARGET_DATASET_NAME "C7" \
               --OUTPUT_PATH "/path/to/save/results" \
               --TRAIN_DATASETS_NAMES "['AD2', 'ADx1']" \
               --NUM_FEATURES 5000 \
               --MIN_CELL_TYPE_POPULATION 100 \
               --PRETRAINED_MODEL_PATH "/path/to/pretrained_model_folder" \
               --USE_GPU True
```

### Option 2: Submit a Job in a HPC
If the number of training datasets or the total number of cells is high, we recommend submitting the job using the provided `main.sh` script from the cluster directory. 
This script is adapted to Slurm, but can be easily modified to work on SGE. 
The specific parameters should be adapted depending on the specifications of the HPC.

```bash
cd cluster
sbatch main.sh
```
### Option 3: Running with Docker
You can run JIND-Multi using Docker with the following steps. **You must run these commands with administrator rights**.

#### Option 3.1: Using a pre-built Docker image

1. Pull the pre-built Docker image:

    ```bash
    docker pull xgarrotesan/jind_multi
    ```

2. Run the Docker container, replacing `<PATH>` with the absolute path to the folder on your system that contains the JIND-Multi repository and the `.h5ad` data files:

    ```bash
    docker run -it -v <PATH>:/app xgarrotesan/jind_multi
    ```

   **Important**: The `<PATH>` you map to the container must contain both:
   - The **JIND-Multi repository** (the project files) 
   - The **`.h5ad` data files** you want to process.

3. Activate the Conda environment inside the container:
    ```bash
    conda activate jind
    ```

4. Run JIND-Multi as usual, defining the path by mapping the unit to `app`, which is the container's folder:

    ```json
    {
        "PATH": "/app/pancreas.h5ad",
        "BATCH_COL": "batch",
        "LABELS_COL": "celltype",
        "SOURCE_DATASET_NAME": "0",
        "TARGET_DATASET_NAME": "3",
        "OUTPUT_PATH": "/app/results",
        "TRAIN_DATASETS_NAMES": "['0', '1', '2']", 
        "NUM_FEATURES": 5000,
        "MIN_CELL_TYPE_POPULATION": 5,
        "USE_GPU": true
    }
    ```

5. Finally, start JIND-Multi using the following command:

    ```bash
    run-jind-multi --config "/app/config.json"
    ```

#### Option 3.2: Building the Docker image locally

If you prefer to build the Docker image locally using the provided Dockerfile:

<!-- 1. Clone the repository if you haven't already:
  ```bash
  git clone https://github.com/ML4BM-Lab/JIND-Multi.git
  cd JIND-Multi -->
  
1. Build the Docker image locally:
  ```bash
    docker build -t jind_multi_local .  
  ```

2. Run the Docker container, ensuring that you map the local path to the folder containing both the repository and .h5ad files. Replace <PATH> with the absolute path to your system's directory:

  ```bash
    docker run -it -v <PATH>:/app jind_multi_local
  ```

Then, repeat steps 3., 4. & 5.

### Output
In the `OUTPUT_PATH`, the following outputs are saved:

- A table with the predictions for each cell on the target data (**predicted_label_test_data.xlsx**:), indicating for each cell the probability calculated by the model for each cell type. The `raw_predictions` column shows the cell type with the highest probability before applying the cell type-specific threshold, and the predictions column shows the predicted cell type after filtering.
The final trained models for each annotated batch are stored `.pt` format (saved in `trained_models` folder), and a `target.pth` file with the trained model for the target batch. The file `val_stats_trained_model.json` contains predictions on the validation test set used to compute the thresholds.

- The model performance results on the source batch, intermediate datasets, and validation set after training the classifier and the fine-tuning steps. These results include confusion matrices indicating the number of cells, how many cells were assigned as ___Unknown___, and how many were correctly predicted with the accuracy percentages before (raw) and after applying the threshold (eff), as well as the incorrect predictions and the mean average precision (mAP) per cell type.
For the source batch, confusion matrices are shown after training the classifier and the fine-tuning process. For the intermediate batches, results are shown before aligning the samples to the latent space of the source ("initial"), after alignment ("adapt"), and after the fine-tuning of the classifier and encoder to evaluate the batch aligmnent. If the target batch has labels, for performance pourposes, confusion matrices are also provided before and after the batch removal process, and after fine-tuning classifier using the most confident cells. The history of these confusion matrices is also saved in a PDF file named train[source_batch_name, number_inter_batches]-test[target_batch_name].pdf.

- By default, the user will also have t-SNE/UMAPS plots with projections colored by batch or labels of the latent space generated by passing the samples through the encoder at four different moments:

1) After training the encoder and classifier with the source batch ("after_train_classifier").
2) After removing the batch effect from each intermediate batch and before inferring on the target batch ("initial").
3) After aligning the target batch to the latent code of the source with the GAN training.
4) After tuning the encoder and classifier for the target batch.

## Notebooks
In the `notebooks` folder, there is an example of executing JIND-Multi, explaining in detail the data processing and the internal functioning of the method.

# Compare JIND Methods
To compare the annotation performance of JIND, JIND-Multi, and JIND-Combined (which merges all annotated datasets into a single one without correcting for batch effects) on any target dataset with known true labels, you have two options:

* Run the Python script directly.
* Submit a job to an HPC queue: you can execute the compare_methods.sh script.

```bash

DATA_PATH="/path/to/data/pancreas/pancreas.h5ad"
BATCH_COL="batch"
LABELS_COL="celltype"
SRC_DATASET="0"
TGT_DATASET="3"
OUT_DIR="/path/to/save/results"
NUM_FEAT=5000
MIN_POP=5
USE_GPU=True 
N_TRIAL=0

echo "Running compare-methods with N_TRIAL: $N_TRIAL"
compare-methods --PATH "$DATA_PATH" \
                --BATCH_COL "$BATCH_COL" \
                --LABELS_COL "$LABELS_COL" \
                --SOURCE_DATASET_NAME "$SRC_DATASET" \
                --TARGET_DATASET_NAME "$TGT_DATASET" \
                --OUTPUT_PATH "$OUT_DIR" \
                --NUM_FEATURES "$NUM_FEAT" \
                --MIN_CELL_TYPE_POPULATION "$MIN_POP" \
                --N_TRIAL "$N_TRIAL" \
                --USE_GPU "$USE_GPU"

```
where,
- **`N_TRIAL`**: (int) A numeric identifier assigned to the experiment.

## Input Arguments Information

| Dataset        | Type       | File                                      | BATCH_COL      | LABELS_COL                | SOURCE_DATASET_NAME | TARGET_DATASET_NAME | TRAIN_DATASETS_NAMES                                                                 | MIN_CELL_TYPE_POPULATION |
|----------------|------------|-------------------------------------------|----------------|---------------------------|---------------------|---------------------|-------------------------------------------------------------------------------------|--------------------------|
| Pancreas       | scRNA-seq   | "pancreas.h5ad"                           | "batch"        | "celltype"                 | "0"                 | "3"                 | "['0', '1', '2']"                                                                    | 5                        |
| NSCLC Lung     | scRNA-seq   | "NSCLC_lung_NORMALIZED_FILTERED.h5ad"     | "Donor"        | "predicted_labels_majority"| "Donor 5"           | "Donor 2"           | "['Donor 0', 'Donor 1', 'Donor 3', 'Donor 4', 'Donor 6']"                            | 20                       |
| Neurips Brain  | scRNA-seq   | "All_human_brain.h5ad"                    | "batch"        | "label"                    | "C4"                | "C7"                | "['AD2', 'ADx1', 'ADx2', 'ADx4']"                                                    | 100                      |
| BMMC           | scATAC-seq  | "data_multiome_annotated_BMMC_ATAC.h5ad"  | "batch"        | "cell_type"                | "s4d8"              | "s3d3"              | "['s1d1', 's1d2', 's1d3', 's2d1', 's2d4', 's2d5', 's3d10', 's4d1']"                 | 18                       |


# Additional Information
In the ./jind_multi folder, you will find an extra README that provides a detailed explanation of each of the Python scripts in the `jind_multi` package.



