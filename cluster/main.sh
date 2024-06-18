#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=test
#SBATCH --job-name=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=50gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/scratch/jsanchoz/JIND-Multi/logs/main.out
#SBATCH --mail-type=END
#SBATCH --mail-user=jsanchoz@unav.es
###SBATCH --gres=gpu:rtx3090:1

# Script directory
SCRIPT_DIR="$PWD"

module load Python
source activate /home/jsanchoz/.conda/envs/jind

# Arguments for the Python script
# PANCREAS
DATA_PATH="../resources/data/pancreas/pancreas.h5ad"  # Example value, replace with your desired h5ad data path
BATCH_COL="batch"
LABELS_COL="celltype"
SOURCE_DATASET_NAME="0"
TARGET_DATASET_NAME="3"
OUTPUT_PATH="$SCRIPT_DIR/../output/pancreas"
TRAIN_DATASETS_NAMES="['0', '1', '2']"
NUM_FEATURES=5000
MIN_CELL_TYPE_POPULATION=5

# BRAIN_SCATLAS_ATAC
# DATA_PATH="../resources/data/brain_scatlas_atac/Integrate_Brain_norm_ctrl_caud.h5ad"  # Example value, replace with your desired h5ad data path
# BATCH_COL="SAMPLE_ID"
# LABELS_COL="peaks_snn_res.0.3"
# SOURCE_DATASET_NAME="scATAC_CTRL_CAUD_06_0615_BRAIN"
# TARGET_DATASET_NAME="scATAC_CTRL_CAUD_14_1018_BRAIN"
# OUTPUT_PATH="$SCRIPT_DIR/../output/brain_scatlas_atac"
# TRAIN_DATASETS_NAMES="['scATAC_CTRL_CAUD_06_0615_BRAIN', 'scATAC_CTRL_CAUD_09_1589_BRAIN']"
# NUM_FEATURES=50000
# MIN_CELL_TYPE_POPULATION=100

# Display the arguments before executing
echo "Running Python script with the following parameters:"
echo "DATA_PATH: $DATA_PATH"
echo "BATCH_COL: $BATCH_COL"
echo "LABELS_COL: $LABELS_COL"
echo "SOURCE_DATASET_NAME: $SOURCE_DATASET_NAME"
echo "TARGET_DATASET_NAME: $TARGET_DATASET_NAME"
echo "OUTPUT_PATH: $OUTPUT_PATH"
echo "TRAIN_DATASETS_NAMES: $TRAIN_DATASETS_NAMES"
echo "NUM_FEATURES: $NUM_FEATURES"
echo "MIN_CELL_TYPE_POPULATION: $MIN_CELL_TYPE_POPULATION"

# Execute the Python script
python -u "$SCRIPT_DIR/../main/Main.py" \
    --PATH "$DATA_PATH" \
    --BATCH_COL "$BATCH_COL" \
    --LABELS_COL "$LABELS_COL" \
    --SOURCE_DATASET_NAME "$SOURCE_DATASET_NAME" \
    --TARGET_DATASET_NAME "$TARGET_DATASET_NAME" \
    --OUTPUT_PATH "$OUTPUT_PATH" \
    --TRAIN_DATASETS_NAMES "$TRAIN_DATASETS_NAMES" \
    --NUM_FEATURES "$NUM_FEATURES" \
    --MIN_CELL_TYPE_POPULATION "$MIN_CELL_TYPE_POPULATION"
