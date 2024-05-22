from Utils import find_saved_models, load_trained_models, load_val_stats
from JindWrapper import JindWrapper
from DataLoader import load_and_process_data
from ConfigLoader import get_config
import argparse

def main(args): 
    # 0) Setting the training configuration (you can modify more things here)
    config = get_config()
    config['data']['num_features'] = args.NUM_FEATURES
    config['data']['min_cell_type_population'] = args.MIN_CELL_TYPE_POPULATION
    
    # 1) Load data and normalized
    data = load_and_process_data(args.PATH, args.BATCH_COL, args.LABELS_COL, config) 

    # 2) Divide in train and test
    train_data = data[data['batch'] != args.TARGET_DATASET_NAME]
    test_data = data[data['batch'] == args.TARGET_DATASET_NAME]

    # 3) Create the Jind Multi object
    jind = JindWrapper(
                        train_data=train_data, 
                        train_dataset_names = args.TRAIN_DATASETS_NAMES, # es una lista,
                        source_dataset_name=args.SOURCE_DATASET_NAME, 
                        output_path=args.OUTPUT_PATH
                    )
                        
    # 4) Train Jind Multi
    # Check if there is already a trained model available
    file_paths = find_saved_models(args.OUTPUT_PATH, train_data)
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
        print('[main] Warning: Trained JIND Multi with this data for the first time')
        jind.train(target_data = test_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main script to execute JindMulti and annotate a target batch using several annotated batches')
    parser.add_argument('--PATH', type=str, required=True, help='Path to the ann object dataset file with gene expression data of shape cells_id x genes')
    parser.add_argument('--BATCH_COL', type=str, required=True, help='Name of the batch column')
    parser.add_argument('--LABELS_COL', type=str, required=True, help='Name of the labels column')
    parser.add_argument('--SOURCE_DATASET_NAME', type=str, required=True, help='Name or ID of the source batch') 
    parser.add_argument('--TARGET_DATASET_NAME', type=str, required=True, help='Name or ID of the target batch')
    parser.add_argument('--OUTPUT_PATH', type=str, required=True, help='Output path to save results and trained model')
    parser.add_argument('--TRAIN_DATASETS_NAMES', type=str, nargs='*', default=None, help='Optional. List of training batch names in desired order, starting with the source batch, followed by intermediate batches in the order they should be processed')
    parser.add_argument('--NUM_FEATURES', type=int, default=5000, help='Optional. Number of genes to consider for modelling, for default is 5000')
    parser.add_argument('--MIN_CELL_TYPE_POPULATION', type=int, default=100, help='Optional. For each batch, the minimum number of cells per cell type necessary for modeling. If this requirement is not met in any batch, the samples belonging to this cell type are removed from all batches')

    args = parser.parse_args()
    main(args)

# pancreas:  python Main.py --PATH '../resources/data/pancreas/pancreas.h5ad' --BATCH_COL 'batch' --LABELS_COL 'celltype' --SOURCE_DATASET_NAME 0 --TARGET_DATASET_NAME 3 --OUTPUT_PATH '../output/pancreas' --TRAIN_DATASETS_NAMES 0 1 2 --NUM_FEATURES 5000 --MIN_CELL_TYPE_POPULATION 5 
# brain_scatlas_atac: python Main.py --PATH '../resources/data/brain_scatlas_atac/Integrated_Brain_norm.h5ad' --BATCH_COL 'SAMPLE_ID' --LABELS_COL 'peaks_snn_res.0.3' --SOURCE_DATASET_NAME   --TARGET_DATASET_NAME  --OUTPUT_PATH '../output/-' --TRAIN_DATASETS_NAMES --NUM_FEATURES 10000 --MIN_CELL_TYPE_POPULATION 100

