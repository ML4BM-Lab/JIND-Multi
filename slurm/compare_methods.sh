#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=compare_methods
#SBATCH --cpus-per-task=4
#SBATCH --mem=500gb
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/scratch/your_username/JIND-Multi/logs/compare_methods_%j.out # Update this path
#SBATCH --mail-type=END
#SBATCH --mail-user=your_email@example.com  # Update this email address

# Script directory
SCRIPT_DIR="$PWD"

module load Python
source activate /path/to/your/environment  # Insert the path to your environment

# Arguments for the Python script
DATA_PATH="../resources/data/pancreas/pancreas.h5ad"  # Example value, replace with your desired h5ad data path
BATCH_COL="batch"
LABELS_COL="celltype"
SOURCE_DATASET_NAME="0"
TARGET_DATASET_NAME="3"     
OUTPUT_PATH="$SCRIPT_DIR/../output/pancreas"
NUM_FEATURES=5000
MIN_CELL_TYPE_POPULATION=5
USE_GPU=True  

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
  echo "USE_GPU: $USE_GPU"

  # Execute the Python script
  compare-methods --PATH "$DATA_PATH" \
                  --BATCH_COL "$BATCH_COL" \
                  --LABELS_COL "$LABELS_COL" \
                  --SOURCE_DATASET_NAME "$SOURCE_DATASET_NAME" \
                  --TARGET_DATASET_NAME "$TARGET_DATASET_NAME" \
                  --OUTPUT_PATH "$OUTPUT_PATH" \
                  --N_TRIAL "$N_TRIAL" \
                  --NUM_FEATURES "$NUM_FEATURES" \
                  --MIN_CELL_TYPE_POPULATION "$MIN_CELL_TYPE_POPULATION" \
                  --USE_GPU "$USE_GPU"
done
