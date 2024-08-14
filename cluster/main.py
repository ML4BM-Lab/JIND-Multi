import argparse
import ast
from jind_multi.core import run_main
from jind_multi.config_loader import get_config

def parse_args():
    parser = argparse.ArgumentParser(description='Main script to execute JindMulti and annotate a target batch using several annotated batches')
    parser.add_argument('--PATH', type=str, required=True, help='Path where you store the `.h5ad` file with your data')
    parser.add_argument('--BATCH_COL', type=str, required=True, help='Name of the column with the information of the different batches or donors in your AnnData object')
    parser.add_argument('--LABELS_COL', type=str, required=True, help='Name of the column with the different cell types in your AnnData object')
    parser.add_argument('--SOURCE_DATASET_NAME', type=str, default=None, help='Optional. name of the source batch. Alternatively, if no batch is specified, JIND-Multi will select as source the batch that produces the least amount of rejected cells on the target batch when used as source in JIND (i.e., without additional intermediate batches)') 
    parser.add_argument('--TARGET_DATASET_NAME', type=str, required=True, help='Name of the target batch')
    parser.add_argument('--OUTPUT_PATH', type=str, required=True, help='Output path to save results and trained model')
    parser.add_argument('--PRETRAINED_MODEL_PATH', type=str, default="", help='Optional. Path to a folder containing pre-trained models. If this path is provided, the script will use the models from this folder instead of training new ones. The folder should contain model files that are compatible with the script\'s requirements. If this argument is not provided or left empty, the script will proceed to train a new model from scratch based on the provided data.')
    parser.add_argument('--TRAIN_DATASETS_NAMES', type=str, default=None, help='Optional. This setting allows you to specify the order of intermediate datasets used for training. You do not need to include the source dataset name here, as it will be selected automatically. If you leave this option empty, the system will use the intermediate datasets in the order they appear in the data')
    parser.add_argument('--NUM_FEATURES', type=int, default=5000, help='Optional. Number of genes to consider for modeling, default is 5000')
    parser.add_argument('--MIN_CELL_TYPE_POPULATION', type=int, default=100, help='Optional. For each batch, the minimum number of cells per cell type necessary for modeling. If this requirement is not met in any batch, the samples belonging to this cell type are removed from all batches')
    parser.add_argument('--USE_GPU', type=ast.literal_eval, default=True, help='Optional. Use CUDA if available (True/False), default is True')
    return parser.parse_args()

def main():
    args = parse_args()
    config = get_config()  # Load configuration
    run_main(args, config)  # Run the main logic

if __name__ == '__main__':
    main()
 