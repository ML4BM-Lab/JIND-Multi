#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=200gb
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/scratch/your_username/JIND-Multi/logs/main.out  # Update this path
#SBATCH --mail-type=END
#SBATCH --mail-user=your_email@example.com  # Update this email address

# Script directory
SCRIPT_DIR="$PWD"

module load Python
source activate /path/to/your/environment  # Insert the path to your environment

# NEURIPS
DATA_PATH="../resources/data/human_brain/All_human_brain.h5ad"  # path to your data 
BATCH_COL="batch"
LABELS_COL="label"
SOURCE_DATASET_NAME="C4"
TARGET_DATASET_NAME="C7"
INTER_DATASETS_NAMES="['AD2', 'ADx1', 'ADx2', 'ADx4']" 
NUM_FEATURES=5000
MIN_CELL_TYPE_POPULATION=100
OUTPUT_PATH="$SCRIPT_DIR/../results/brain_neurips"  # Path where you want to save the results
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
echo "INTER_DATASETS_NAMES: $INTER_DATASETS_NAMES"
echo "NUM_FEATURES: $NUM_FEATURES"
echo "MIN_CELL_TYPE_POPULATION: $MIN_CELL_TYPE_POPULATION"
echo "PRETRAINED_MODEL_PATH: $PRETRAINED_MODEL_PATH"
echo "USE_GPU: $USE_GPU"

if [ -z "$PRETRAINED_MODEL_PATH" ]; then
    echo "PRETRAINED_MODEL_PATH is empty. Running without pre-trained model."
    run-jind-multi --PATH "$DATA_PATH" \
                   --BATCH_COL "$BATCH_COL" \
                   --LABELS_COL "$LABELS_COL" \
                   --SOURCE_DATASET_NAME "$SOURCE_DATASET_NAME" \
                   --TARGET_DATASET_NAME "$TARGET_DATASET_NAME" \
                   --OUTPUT_PATH "$OUTPUT_PATH" \
                   --INTER_DATASETS_NAMES "$INTER_DATASETS_NAMES" \
                   --NUM_FEATURES "$NUM_FEATURES" \
                   --MIN_CELL_TYPE_POPULATION "$MIN_CELL_TYPE_POPULATION" \
                   --USE_GPU "$USE_GPU"
else
    echo "Using pre-trained model from $PRETRAINED_MODEL_PATH"
    run-jind-multi --PATH "$DATA_PATH" \
                   --BATCH_COL "$BATCH_COL" \
                   --LABELS_COL "$LABELS_COL" \
                   --SOURCE_DATASET_NAME "$SOURCE_DATASET_NAME" \
                   --TARGET_DATASET_NAME "$TARGET_DATASET_NAME" \
                   --OUTPUT_PATH "$OUTPUT_PATH" \
                   --INTER_DATASETS_NAMES "$INTER_DATASETS_NAMES" \
                   --NUM_FEATURES "$NUM_FEATURES" \
                   --MIN_CELL_TYPE_POPULATION "$MIN_CELL_TYPE_POPULATION" \
                   --PRETRAINED_MODEL_PATH "$PRETRAINED_MODEL_PATH" \
                   --USE_GPU "$USE_GPU"
fi

    
