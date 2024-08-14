# JIND-Multi  
<!-- #### Publication:   -->
We introduce JIND-Multi, an extension of the JIND framework for the automated annotation of single-cell RNA sequencing (scRNA-Seq) data ([Goyal et al., 2022](https://academic.oup.com/bioinformatics/article/38/9/2488/6543609)), that allows transferring cell-type labels from several annotated datasets. Notably, JIND-Multi is also applicable for the annotation of scATAC-Seq data. Moreover, similarly to its predecessor, JIND-Multi has the option to mark cells as "unassigned" if the model does not produce reliable predictions, i.e., below some pre-computed cell type specific thresholds.

The proposed approach JIND-Multi can leverage a large number of annotated datasets, e.g., those that compose an atlas, making the annotation of unlabeled datasets more precise and with lower rejection rates (unassigned cells). We provide an efficient implementation of JIND-Multi that is publicly available and ready to use by the community.

<p align="center">
    <img src="https://github.com/ML4BM-Lab/JIND-Multi/blob/master/JIND.png" width="700">
</p>

## Prerequisites
- Linux or macOS
- Miniconda
- Python 3.6 or higher (tested on 3.6.8 and 3.7.11)
- CPU or NVIDIA GPU + CUDA CuDNN

## Installation
```bash
git clone https://github.com/ML4BM-Lab/JIND-Multi.git
cd JIND
conda create -n jind python=3.7.16 
conda activate jind
pip install -e .
```

## Data
The datasets to reproduce the results presented in the manuscript are available at the following link:  https://doi.org/10.5281/zenodo.11098805

# Executing JIND-Multi
There are two options to execute the JIND-Multi framework: 
* Running the Python script 
* Submitting a job to a HPC queue (recommended for reproducing the 10 fold cross validations from the manuscript)

### Option 1: The Python Script 
For executing JIND-Multi on the `Brain Neurips` dataset, we would do it as follows:

```bash
run-jind-multi --PATH /path/to/data/All_human_brain.h5ad \
               --BATCH_COL "batch" \
               --LABELS_COL "label" \
               --SOURCE_DATASET_NAME "C4" \
               --TARGET_DATASET_NAME "C7" \
               --OUTPUT_PATH /path/to/save/results \
               --TRAIN_DATASETS_NAMES "['AD2', 'ADx1']" \
               --NUM_FEATURES 5000 \
               --MIN_CELL_TYPE_POPULATION 100 \
               --PRETRAINED_MODEL_PATH /path/to/pretrained_model_folder \
               --USE_GPU True
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

### Option 2: Submit a Job in a HPC
If the number of training datasets or the total number of cells is high, we recommend submitting the job using the provided `main.sh` script from the cluster directory. 
This script is adapted to Slurm, but can be easily modified to work on SGE. 
The specific parameters should be adapted depending on the specifications of the HPC.

```bash
cd cluster
sbatch main.sh
```

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
To compare the annotation performance of JIND, JIND-Multi, and JIND-Combined (merging all annotated datasets into one, without correcting for batch effect) for any target dataset with known true labels, you can execute the `compare_methods.sh` script.

# Additional Information
In the `./jind_multi` folder, you will find an extra README explaining in more detail each of the Python scripts of the jind_multi package.





