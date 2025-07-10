
# 🔧 Executing JIND-Multi

**JIND-Multi** can be executed in three different **modes**, depending on your computational resources and preferences:

1. **Locally**, using the Python script.
2. On a **High-Performance Computing (HPC)** cluster.
3. Inside a **Docker container**, either locally or through the provided web tool.

---

## ⚙️ Installation (Required for Local & HPC Execution)

Before running JIND-Multi locally or in an HPC environment, install the framework as follows:

```bash
git clone https://github.com/ML4BM-Lab/JIND-Multi.git
cd JIND-Multi
conda create -n jind python=3.7.16 
conda activate jind
pip install -e .
```

## 🖥️ Option 1: Local Execution via Python Script

### ✅ Using a JSON configuration file

Create a configuration file (`config.json`) with the following example content:

```json
{
  "PATH": "/path/to/data/All_human_brain.h5ad",
  "BATCH_COL": "batch",
  "LABELS_COL": "label",
  "SOURCE_DATASET_NAME": "C4",
  "TARGET_DATASET_NAME": "C7",
  "OUTPUT_PATH": "/path/to/save/results",
  "INTER_DATASETS_NAMES": "['ADx1']",
  "EXCLUDE_DATASETS_NAMES": "['AD2']",
  "NUM_FEATURES": 5000,
  "MIN_CELL_TYPE_POPULATION": 100,
  "PRETRAINED_MODEL_PATH": "/path/to/pretrained_model_folder",
  "USE_GPU": true
}
```

Then run:

```bash
run-jind-multi --config /path/to/config.json
```

ℹ️ For a full explanation of the arguments, refer to [Required Inputs & Configuration Options](README_general.md#-required-inputs--configuration-options)


### ✅ Running directly from the command line

```bash
run-jind-multi --PATH "/path/to/data/All_human_brain.h5ad" \
               --BATCH_COL "batch" \
               --LABELS_COL "label" \
               --SOURCE_DATASET_NAME "C4" \
               --TARGET_DATASET_NAME "C7" \
               --OUTPUT_PATH "/path/to/save/results" \
               --INTER_DATASETS_NAMES "['ADx1']" \
               --EXCLUDE_DATASETS_NAMES "['AD2']" \
               --NUM_FEATURES 5000 \
               --MIN_CELL_TYPE_POPULATION 100 \
               --PRETRAINED_MODEL_PATH "/path/to/pretrained_model_folder" \
               --USE_GPU True
```

## 🧬 Option 2: Execution on HPC (SLURM)

If you are working with large datasets, we recommend running JIND-Multi in an HPC environment.

1. Go to the `cluster/` directory:

```bash
cd cluster
```

2. Submit the job using SLURM:

```bash
sbatch main.sh
```

## 🐳 Option 3: Docker Execution
If you prefer to avoid local installations, you can run JIND-Multi inside a Docker container.

#### Option 3.1: Using a pre-built Docker image

1. Pull the pre-built Docker image:

    ```bash
    docker pull ju4n5ilva/jind_multi
    ```

2. Run the Docker container, replacing `<PATH>` with the absolute path to the folder on your system that contains the JIND-Multi repository and the `.h5ad` data files:

    ```bash
    docker run -it -p 5003:5003 -v <PATH>:/app ju4n5ilva/jind_multi
    ```
You can also integrate it with the JIND-Multi WebTool for an even smoother experience.
- to run with webTool select opcion 1 
- to run in bash select opcion 2
#### Option 3.2: Building the Docker image locally

If you prefer to build the Docker image locally using the provided Dockerfile:

<!-- 1. Clone the repository if you haven't already:
  ```bash
  git clone https://github.com/ML4BM-Lab/JIND-Multi.git
  cd JIND-Multi -->

0. Go to repository folder absolute path in our case <PATH>
  ```bash
    cd <PATH>  
  ```
1. Build the Docker image locally:
  ```bash
    docker build --no-cache -t jind_multi:latest . 
  ```

2. Run the Docker container, ensuring that you map the local path to the folder containing both the repository and .h5ad files. Replace <PATH> with the absolute path to your system's directory:

  ```bash
    docker run -it -p 5003:5003 -v <PATH>:/app jind_multi
  ```

You can also integrate it with the JIND-Multi WebTool for an even smoother experience.

- to run with webTool select opcion 1 
- to run in bash select opcion 2

# Compare JIND Methods

This guide explains how to evaluate and compare the annotation performance of **JIND**, **JIND-Multi**, and **JIND-Combined** (a method that merges all annotated datasets into one without batch effect correction) on any target dataset with known true labels.

You have two options to run the comparison:
- Run the Python script directly.
- Submit a job to an HPC queue by executing the `compare_methods.sh` script.

## Example Command

The following example demonstrates how to run the `compare-methods` script with command-line arguments:

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

Alternatively, you can also run `compare-methods` using a configuration file in JSON format:

```bash
compare-methods --config /path/to/config.json
```

## Contact
For questions, feedback, or support, please contact:  
**Joseba Sancho-Zamora**  
Email: [jsanchoz@unav.es](mailto:jsanchoz@unav.es)

