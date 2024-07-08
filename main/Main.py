from Utils import find_saved_models, load_trained_models, load_val_stats
from JindWrapper import JindWrapper
from DataLoader import load_and_process_data
from ConfigLoader import get_config
import argparse
import ast

def main(args, config):
    # 1) Load data and normalize
    # data = load_and_process_data(args.PATH, args.BATCH_COL, args.LABELS_COL, config) 
    data = load_and_process_data(args, config) 

    # 2) Divide into train and test
    train_data = data[data['batch'] != args.TARGET_DATASET_NAME]
    test_data = data[data['batch'] == args.TARGET_DATASET_NAME]

    # 3) Create the Jind Multi object
    if args.TRAIN_DATASETS_NAMES:
        train_datasets_names = ast.literal_eval(args.TRAIN_DATASETS_NAMES) # If it's not None, parse it as a list
    else:
        train_datasets_names = args.TRAIN_DATASETS_NAMES   

    jind = JindWrapper(
                        train_data=train_data, 
                        train_dataset_names=train_datasets_names,  
                        source_dataset_name=args.SOURCE_DATASET_NAME, 
                        output_path=args.OUTPUT_PATH,
                        config = config
                    )
                        
    # 4) Train Jind Multi
    file_paths = find_saved_models(args.OUTPUT_PATH, train_data) # Check if there is already a trained model available
    if file_paths:
        print('[main] Warning: Trained Models detected')
        print(f'[main] File Paths: {file_paths}')
        # Load the trained models
        print("[main] e.2) Load the trained models")
        model = load_trained_models(file_paths, train_data, args.SOURCE_DATASET_NAME)
        # Load the val_stats
        print("[main] e.3) Load the val_stats")
        val_stats = load_val_stats(args.OUTPUT_PATH, 'val_stats_trained_model.json') 
        # Do Jind
        print("[main] f) Do Jind")
        jind.train(target_data=test_data, model=model, val_stats=val_stats)

    else:
        print('[main] Warning: Training JIND Multi with this data for the first time')
        jind.train(target_data=test_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main script to execute JindMulti and annotate a target batch using several annotated batches')
    parser.add_argument('--PATH', type=str, required=True, help='Path where you store the `.h5ad` file with your data')
    parser.add_argument('--BATCH_COL', type=str, required=True, help='Name of the column with the information of the different batches or donors in your AnnData object')
    parser.add_argument('--LABELS_COL', type=str, required=True, help='Name of the column with the different cell types in your AnnData object')
    parser.add_argument('--SOURCE_DATASET_NAME', type=str, default=None, help='Optional. name of the source batch. Alternatively, if no batch is specified, JIND-Multi will select as source the batch that produces the least amount of rejected cells on the target batch when used as source in JIND (i.e., without additional intermediate batches)') 
    parser.add_argument('--TARGET_DATASET_NAME', type=str, required=True, help='Name of the target batch')
    parser.add_argument('--OUTPUT_PATH', type=str, required=True, help='Output path to save results and trained model')
    parser.add_argument('--TRAIN_DATASETS_NAMES', type=str, default=None, help='Optional. This setting allows you to specify the order of intermediate datasets used for training. You do not need to include the source dataset name here, as it will be selected automatically. If you leave this option empty, the system will use the intermediate datasets in the order they appear in the data')
    parser.add_argument('--NUM_FEATURES', type=int, default=5000, help='Optional. Number of genes to consider for modeling, default is 5000')
    parser.add_argument('--MIN_CELL_TYPE_POPULATION', type=int, default=100, help='Optional. For each batch, the minimum number of cells per cell type necessary for modeling. If this requirement is not met in any batch, the samples belonging to this cell type are removed from all batches')
    parser.add_argument('--USE_CUDA', action='store_false', help='Optional. Set this flag to disable CUDA, default is True (enabled)') # Eliminate this line JOSEBA
    args = parser.parse_args()

    # 0) Setting the training configuration (you can modify more things here)
    config = get_config()
    config['data']['num_features'] = args.NUM_FEATURES
    config['data']['min_cell_type_population'] = args.MIN_CELL_TYPE_POPULATION
    config['train_classifier']['cuda'] = args.USE_CUDA  
    config['GAN']['cuda'] = args.USE_CUDA  
    config['ftune']['cuda'] = args.USE_CUDA  
    print(f'use_cuda: {args.USE_CUDA}')
    main(args, config)

