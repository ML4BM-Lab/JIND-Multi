import argparse
import ast
from jind_multi.config_loader import get_config
from jind_multi.utils import load_config_from_file, find_saved_models, load_trained_models, load_val_stats
from jind_multi.jind_wrapper import JindWrapper
from jind_multi.data_loader import load_and_process_data

def parse_args():
    parser = argparse.ArgumentParser(description='Main script to execute JindMulti and annotate a target batch using several annotated batches')
    parser.add_argument('--config', type=str, help='Path to JSON config file')
    parser.add_argument('--PATH', type=str, help='Path where you store the `.h5ad` file with your data')
    parser.add_argument('--BATCH_COL', type=str, help='Name of the column with the information of the different batches or donors in your AnnData object')
    parser.add_argument('--LABELS_COL', type=str, help='Name of the column with the different cell types in your AnnData object')
    parser.add_argument('--SOURCE_DATASET_NAME', type=str, default=None, help='Optional. Name of the source batch. Alternatively, if no batch is specified, JIND-Multi will select as source the batch that produces the least amount of rejected cells on the target batch when used as source in JIND (i.e., without additional intermediate batches)')
    parser.add_argument('--TARGET_DATASET_NAME', type=str, help='Name of the target batch')
    parser.add_argument('--OUTPUT_PATH', type=str, help='Output path to save results and trained model')
    parser.add_argument('--PRETRAINED_MODEL_PATH', type=str, default="", help='Optional. Path to a folder containing pre-trained models. If this path is provided, the script will use the models from this folder instead of training new ones. The folder should contain model files that are compatible with the script\'s requirements. If this argument is not provided or left empty, the script will proceed to train a new model from scratch based on the provided data.')
    parser.add_argument('--INTER_DATASETS_NAMES', type=str, default=None, help='Optional. A comma-separated list of dataset names. This setting allows you to specify the order of intermediate datasets used for training. You do not need to include the source dataset name here, as it will be selected automatically. If you leave this option empty, the system will use the intermediate datasets in the order they appear in the data')
    parser.add_argument('--EXCLUDE_DATASETS_NAMES', type=str, default=None, help='Optional. A comma-separated list of dataset names to exclude from training.')
    parser.add_argument('--NUM_FEATURES', type=int, default=5000, help='Optional. Number of genes to consider for modeling, default is 5000')
    parser.add_argument('--MIN_CELL_TYPE_POPULATION', type=int, default=100, help='Optional. For each batch, the minimum number of cells per cell type necessary for modeling. If this requirement is not met in any batch, the samples belonging to this cell type are removed from all batches')
    parser.add_argument('--USE_GPU', type=ast.literal_eval, default=True, help='Optional. Use CUDA if available (True/False), default is True')
    return parser.parse_args()

def execute_jind_multi(args, config):
    """Execute the JIND Multi logic."""
    # Load data and normalize
    data = load_and_process_data(args, config)

    # Divide into train and test datasets
    train_data = data[data['batch'] != args.TARGET_DATASET_NAME]
    test_data = data[data['batch'] == args.TARGET_DATASET_NAME]

    # Create the Jind Multi object
    jind = JindWrapper(
        train_data=train_data, 
        train_dataset_names=ast.literal_eval(args.INTER_DATASETS_NAMES) if args.INTER_DATASETS_NAMES else None,  
        source_dataset_name=args.SOURCE_DATASET_NAME, 
        output_path=args.OUTPUT_PATH,
        config=config
    )

    # Train Jind Multi
    if args.PRETRAINED_MODEL_PATH:
        print('[main] Loading pre-trained models from specified path')
        file_paths = find_saved_models(args.PRETRAINED_MODEL_PATH, train_data) # Check if there is already a trained model available
        
        if file_paths:
            print('[main] Warning: Trained Models detected')
            print(f'[main] File Paths: {file_paths}')
            # Load the trained models
            print("[main] Load the trained models")
            model = load_trained_models(file_paths, train_data, args.SOURCE_DATASET_NAME)
            # Load the val_stats for classification filtering scheme
            print("[main] Load the val_stats")
            val_stats = load_val_stats(args.PRETRAINED_MODEL_PATH, 'val_stats_trained_model.json') 
            # Train using JIND
            jind.train(target_data=test_data, model=model, val_stats=val_stats)
        
        else:
            print('[main] No pre-trained models found at the specified path. Please check the path and try again.')
            raise FileNotFoundError(f'No pre-trained models found at the specified path: {args.PRETRAINED_MODEL_PATH}. Please check the path and try again.')
    else:
        print('[main] No pre-trained model path provided. Training JIND Multi with the provided data for the first time.')
        jind.train(target_data=test_data)

def main():
    args = parse_args()
    print("\n[INFO] Received arguments:")
    for arg_name, arg_value in vars(args).items():
        print(f"  {arg_name}: {arg_value} (type: {type(arg_value).__name__})")

    # Load configuration from file if specified
    if args.config:
        config_data = load_config_from_file(args.config)
        # Override arguments with values from config file if they are not None
        for key, value in config_data.items():
            if value is not None:
                setattr(args, key, value)

    config = get_config()  # Load configuration
    
    # Set up training configuration (you can modify more things here)
    config['data']['num_features'] = args.NUM_FEATURES
    config['data']['min_cell_type_population'] = args.MIN_CELL_TYPE_POPULATION
    config['train_classifier']['cuda'] = args.USE_GPU  
    config['GAN']['cuda'] = args.USE_GPU  
    config['ftune']['cuda'] = args.USE_GPU  

    # Execute the JIND Multi logic
    execute_jind_multi(args, config)

if __name__ == '__main__':
    main()

 
