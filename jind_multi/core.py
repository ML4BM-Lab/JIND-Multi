import ast
from .utils import find_saved_models, load_trained_models, load_val_stats
from .jind_wrapper import JindWrapper
from .data_loader import load_and_process_data

def run_main(args, config):
    # 1) Load data and normalize
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
                        config=config
                    )
    # 4) Train Jind Multi
    if args.PRETRAINED_MODEL_PATH:
        print('[main] Loading pre-trained models from specified path')
        file_paths = find_saved_models(args.PRETRAINED_MODEL_PATH, train_data) # Check if there is already a trained model available
        if file_paths:
            print('[main] Warning: Trained Models detected')
            print(f'[main] File Paths: {file_paths}')
            # Load the trained models
            print("[main] Load the trained models")
            model = load_trained_models(file_paths, train_data, args.SOURCE_DATASET_NAME)
            # Load the val_stats
            print("[main] Load the val_stats")
            val_stats = load_val_stats(args.PRETRAINED_MODEL_PATH, 'val_stats_trained_model.json') 
            # Do JIND
            jind.train(target_data=test_data, model=model, val_stats=val_stats)
        else:
            print('[main] No pre-trained models found at the specified path. Please check the path and try again.')
            raise FileNotFoundError(f'No pre-trained models found at the specified path: {args.PRETRAINED_MODEL_PATH}. Please check the path and try again.')
    else:
        print('[main] No pre-trained model path provided. Training JIND Multi with the provided data for the first time.')
        jind.train(target_data=test_data)