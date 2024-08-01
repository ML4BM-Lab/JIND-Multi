#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=200gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/scratch/jsanchoz/JIND-Multi/logs/main.out
#SBATCH --mail-type=END
#SBATCH --mail-user=jsanchoz@unav.es
###SBATCH --gres=gpu:rtx3090:1

# Script directory
SCRIPT_DIR="$PWD"

module load Python
source activate /home/jsanchoz/.conda/envs/jind # here insert path to your environment 

# NEURIPS
DATA_PATH="../resources/data/human_brain/All_human_brain.h5ad"  # path to your data 
BATCH_COL="batch"
LABELS_COL="label"
SOURCE_DATASET_NAME="C4"
TARGET_DATASET_NAME="C7"
TRAIN_DATASETS_NAMES="['C4', 'AD2', 'ADx1', 'ADx2', 'ADx4']" 
NUM_FEATURES=5000
MIN_CELL_TYPE_POPULATION=100
OUTPUT_PATH="$SCRIPT_DIR/../results/brain_neurips" # path where you want to save the results
PRETRAINED_MODEL_PATH=""  # Specify the path to the pre-trained model if you want to reuse one
USE_GPU=True  

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
echo "PRETRAINED_MODEL_PATH: $PRETRAINED_MODEL_PATH"
echo "USE_GPU: $USE_GPU"

# Execute the Python script with conditional model path
if [ -z "$PRETRAINED_MODEL_PATH" ]; then
    echo "PRETRAINED_MODEL_PATH is empty"
    # If PRETRAINED_MODEL_PATH is empty, run without the --PRETRAINED_MODEL_PATH argument
    python -u "$SCRIPT_DIR/../main/Main.py" \
        --PATH "$DATA_PATH" \
        --BATCH_COL "$BATCH_COL" \
        --LABELS_COL "$LABELS_COL" \
        --SOURCE_DATASET_NAME "$SOURCE_DATASET_NAME" \
        --TARGET_DATASET_NAME "$TARGET_DATASET_NAME" \
        --OUTPUT_PATH "$OUTPUT_PATH" \
        --TRAIN_DATASETS_NAMES "$TRAIN_DATASETS_NAMES" \
        --NUM_FEATURES "$NUM_FEATURES" \
        --MIN_CELL_TYPE_POPULATION "$MIN_CELL_TYPE_POPULATION" \
        --USE_GPU "$USE_GPU"
else
    # If PRETRAINED_MODEL_PATH is set, pass it to the --PRETRAINED_MODEL_PATH argument
    echo "PRETRAINED_MODEL_PATH is not empty"
    python -u "$SCRIPT_DIR/../main/Main.py" \
        --PATH "$DATA_PATH" \
        --BATCH_COL "$BATCH_COL" \
        --LABELS_COL "$LABELS_COL" \
        --SOURCE_DATASET_NAME "$SOURCE_DATASET_NAME" \
        --TARGET_DATASET_NAME "$TARGET_DATASET_NAME" \
        --OUTPUT_PATH "$OUTPUT_PATH" \
        --TRAIN_DATASETS_NAMES "$TRAIN_DATASETS_NAMES" \
        --NUM_FEATURES "$NUM_FEATURES" \
        --MIN_CELL_TYPE_POPULATION "$MIN_CELL_TYPE_POPULATION" \
        --PRETRAINED_MODEL_PATH "$PRETRAINED_MODEL_PATH"
        --USE_GPU "$USE_GPU" \
fi
    