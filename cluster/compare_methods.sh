#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=compare_methods
#SBATCH --cpus-per-task=4
#SBATCH --mem=500gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/scratch/jsanchoz/JIND-Multi/logs/compare_methods_%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=jsanchoz@unav.es
#--gres=gpu:rtx3090:1

# Script directory
SCRIPT_DIR="$PWD"

# Arguments for the Python script
DATA_PATH="../resources/data/brain_scatlas_atac/Integrate_Brain_norm_ctrl_caud.h5ad"  # Example value, replace with your desired h5ad data path
BATCH_COL="SAMPLE_ID"
LABELS_COL="peaks_snn_res.0.3"
SOURCE_DATASET_NAME="scATAC_CTRL_CAUD_06_0615_BRAIN"
TARGET_DATASET_NAME="scATAC_CTRL_CAUD_14_1018_BRAIN"
OUTPUT_PATH="$SCRIPT_DIR/../output/method_comparation/brain_scatlas_atac"
NUM_FEATURES=50000
MIN_CELL_TYPE_POPULATION=100

# Loop to run the Python script 10 times with different N_TRIAL values
for N_TRIAL in {0..9}
do
  # Display the arguments before executing
  echo "Running Python script with the following parameters:"
  echo "DATA_PATH: $DATA_PATH"
  echo "BATCH_COL: $BATCH_COL"
  echo "LABELS_COL: $LABELS_COL"
  echo "SOURCE_DATASET_NAME: $SOURCE_DATASET_NAME"
  echo "TARGET_DATASET_NAME: $TARGET_DATASET_NAME"
  echo "OUTPUT_PATH: $OUTPUT_PATH"
  echo "N_TRIAL: $N_TRIAL"
  echo "NUM_FEATURES: $NUM_FEATURES"
  echo "MIN_CELL_TYPE_POPULATION: $MIN_CELL_TYPE_POPULATION"

  # Execute the Python script
  python -u "$SCRIPT_DIR/../main/CompareMethods.py" \
    --PATH "$DATA_PATH" \
    --BATCH_COL "$BATCH_COL" \
    --LABELS_COL "$LABELS_COL" \
    --SOURCE_DATASET_NAME "$SOURCE_DATASET_NAME" \
    --TARGET_DATASET_NAME "$TARGET_DATASET_NAME" \
    --OUTPUT_PATH "$OUTPUT_PATH" \
    --N_TRIAL "$N_TRIAL" \
    --NUM_FEATURES "$NUM_FEATURES" \
    --MIN_CELL_TYPE_POPULATION "$MIN_CELL_TYPE_POPULATION"
done
