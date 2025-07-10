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

## Prerequisites for Running Locally

- **Operating System:** Linux or macOS
- **Environment Manager:** [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- **Programming Language:** Python 3.6.8 or higher
- **Hardware:** A CPU is sufficient, but using an NVIDIA GPU with CUDA and cuDNN is recommended for better performance.


## 📁 Required Inputs & Configuration Options

To run JIND-Multi, whether you're training from scratch or using pre-trained models, you need to provide a `.h5ad` file containing your single-cell data and define a few key parameters. Below is a list of required and optional arguments, their types, and what they represent.

| Argument                   | Type     | Required | Description |
|---------------------------|----------|----------|-------------|
| `PATH`                    | `string` | ✅       | Path to the input `.h5ad` file. This file must contain your annotated single-cell dataset (AnnData object). |
| `BATCH_COL`               | `string` | ✅       | Name of the column in `adata.obs` that contains batch or donor identifiers. |
| `LABELS_COL`              | `string` | ✅       | Name of the column in `adata.obs` that contains cell type labels. |
| `TARGET_DATASET_NAME`     | `string` | ✅       | Name of the target batch (from `BATCH_COL`) to which the cell type annotations will be transferred. |
| `SOURCE_DATASET_NAME`     | `string` | ❌       | (Optional) Name of the source batch used for training. If not specified, JIND-Multi will automatically select the best source batch based on rejection rate. |
| `OUTPUT_PATH`             | `string` | ✅       | Path to the directory where output results (metrics, predictions, etc.) will be saved. |
| `PRETRAINED_MODEL_PATH`   | `string` | ❌       | (Optional) Path to a directory with pre-trained `.pt` model files and a `.json` with thresholds. If provided, the model will skip training and proceed directly to inference. |
| `INTER_DATASETS_NAMES`    | `string` | ❌       | (Optional) Comma-separated list of intermediate batch names (from `BATCH_COL`) used in multi-step training. Do **not** include the source batch. |
| `EXCLUDE_DATASETS_NAMES`  | `string` | ❌       | (Optional) Comma-separated list of dataset names to exclude from training. Avoid duplicating entries used in `SOURCE`, `TARGET`, or `INTER`. |
| `NUM_FEATURES`            | `int`    | ❌       | (Optional) Number of genes to include in the model. Default: `5000`. |
| `MIN_CELL_TYPE_POPULATION`| `int`    | ❌       | (Optional) Minimum number of cells per cell type per batch required for training. Default: `100`. |
| `USE_GPU`                 | `bool`   | ❌       | (Optional, but recommended) Set to `True` to train using GPU. Default: `True`. |

### 📌 Notes

- If `PRETRAINED_MODEL_PATH` is provided, JIND-Multi skips training and uses the given models for inference on the target batch.
- If `SOURCE_DATASET_NAME` is not specified, the method automatically selects the source batch that minimizes cell rejection when predicting on the target batch.

## 📤 Output Files and Results Overview

After running JIND-Multi, all outputs are stored in the specified `OUTPUT_PATH` directory. These results include both prediction files and detailed performance metrics for model evaluation.

### 🧪 Prediction Results

- **`predicted_label_test_data.xlsx`**  
  This Excel file contains prediction results for each cell in the **target batch**. For each cell, it includes:
  - Probabilities assigned by the model for each cell type.
  - `raw_predictions`: The cell type with the highest probability (before applying thresholds).
  - `predictions`: The final predicted label after applying cell type-specific thresholds (low-confidence predictions may be marked as _Unknown_).

- **Trained Model Files**  
  - Trained models for each annotated batch are saved in `.pt` format inside the `trained_models/` directory.
  - A separate `target.pth` file contains the model trained on the target batch after fine-tuning.
  - The file `val_stats_trained_model.json` contains the predictions on the validation set used to calculate threshold values per cell type.

### 📊 Performance Metrics and Confusion Matrices

- JIND-Multi evaluates and records the classification performance at every key step:
  - **Source batch**: Confusion matrices before and after fine-tuning.
  - **Intermediate batches**: Performance at three stages:
    - `initial`: Before alignment.
    - `adapt`: After alignment to the source batch.
    - `finetuned`: After final fine-tuning of encoder and classifier.
  - **Target batch** (if labels are available): Confusion matrices showing:
    - Before alignment.
    - After adaptation.
    - After classifier fine-tuning using confident predictions.

- The matrices display:
  - Number of cells per cell type.
  - How many were predicted as _Unknown_.
  - Accuracy before (raw) and after (effective) thresholding.
  - Misclassified cells.
  - Mean Average Precision (mAP) per cell type.

- **Training history summary**  
  A PDF file named `train[SOURCE_BATCH, INTER_COUNT]-test[TARGET_BATCH].pdf` is also generated. It includes a visual history of confusion matrices across training stages, making it easier to interpret performance changes through each step of the pipeline.


## 📂 Datasets and Input Argument Guide

The datasets used to reproduce the results presented in the manuscript are publicly available at the following Zenodo link:  
🔗 https://doi.org/10.5281/zenodo.14000644

> ⚠️ **Important:**  
If you're using any of the datasets from Zenodo, please refer to the [`Input Argument Reference`](#input-argument-reference) below to correctly configure the input arguments when running the method.

---

## 📥 Input Argument Reference

> ⚠️ **High Resource Requirement:**  
The following datasets require a High Performance Computing system (HPC) due to their large size:
- `All_human_brain.h5ad`
- `data_multiome_annotated_BMMC_ATAC.h5ad`

| **Dataset**     | **Type**     | **Filename**                                  | **BATCH_COL** | **LABELS_COL**             | **SOURCE_DATASET_NAME** | **TARGET_DATASET_NAME** | **INTER_DATASETS_NAMES**                                                                 | **MIN_CELL_TYPE_POPULATION** |
|-----------------|--------------|-----------------------------------------------|---------------|-----------------------------|--------------------------|--------------------------|------------------------------------------------------------------------------------------|-------------------------------|
| Pancreas        | scRNA-seq    | `pancreas.h5ad`                               | `batch`       | `celltype`                  | `0`                      | `3`                      | `['1', '2']`                                                                             | 5                             |
| NSCLC Lung      | scRNA-seq    | `NSCLC_lung_NORMALIZED_FILTERED.h5ad`         | `Donor`       | `predicted_labels_majority` | `Donor5`                | `Donor2`                | `['Donor0', 'Donor1', 'Donor3', 'Donor4', 'Donor6']`                                   | 20                            |
| Neurips Brain   | scRNA-seq    | `All_human_brain.h5ad`                        | `batch`       | `label`                     | `C4`                     | `C7`                     | `['AD2', 'ADx1', 'ADx2', 'ADx4']`                                                         | 100                           |
| BMMC            | scATAC-seq   | `data_multiome_annotated_BMMC_ATAC.h5ad`      | `batch`       | `cell_type`                | `s4d8`                   | `s3d3`                   | `['s1d1', 's1d2', 's1d3', 's2d1', 's2d4', 's2d5', 's3d10', 's4d1']`                      | 18                            |
| Fetal Heart     | scATAC-seq   | `heart_sample_norm_scaled_data_annotated.h5ad`| `batch`       | `celltype`                  | `heart_sample_39`        | `heart_sample_14`        | `['heart_sample_32']`                                                                    | 100                           |
| Fetal Kidney    | scATAC-seq   | `kidney_sample_norm_scaled_data_annotated.h5ad`| `batch`      | `celltype`                  | `kidney_sample_3`        | `kidney_sample_67`       | `['kidney_sample_34', 'kidney_sample_65']`                                               | 100                           |

---

## 📁 Additional Documentation
- `README_execution.md`: Located in the **same directory** as this README.  
  It provides step-by-step instructions on how to run the method:
  - ✅ Locally  
  - 🚀 On an HPC system  
  - 🌐 Via the web interface

- Inside the `./jind_multi/` folder:  
  - `README.md`: Contains **detailed documentation** about each Python script included in the `jind_multi` package.

# Inside the `./jind_multi/` folder:  
  - `README_execution.md`: Contains **detailed documentation** about how run the models.

## Contact
For questions, feedback, or support, please contact:  
**Joseba Sancho-Zamora**  
Email: [jsanchoz@unav.es](mailto:jsanchoz@unav.es)