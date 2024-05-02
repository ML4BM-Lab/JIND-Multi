from JindWrapper import JindWrapper
from DataLoader import load_data
import argparse

def main(args): 
    # 1) Load data and normalized    
    data = load_data(data_type = args.DATA_TYPE)

    # 2) Divide in train and test
    train_data = data[data['batch'] != args.TARGET_DATASET_NAME]
    test_data = data[data['batch'] == args.TARGET_DATASET_NAME]

    # 3) Create the Jind Multi object
    jind = JindWrapper(train_data=train_data, source_dataset_name=args.SOURCE_DATASET_NAME, output_path = args.PATH_WD+'/output/'+ args.DATA_TYPE)

    # 4) Train Jind Multi
    # Check if there is already a trained model available
    model_input_path = os.path.abspath(os.path.join(output_path, '../../results')) # esta igual hay que modificar aquí!
    file_paths = find_saved_models(model_input_path, train_data)

    if file_paths:
        print('[main] Warning: Trained Models detected')
        print(f'[main] File Paths: {file_paths}')
        # Load the trained models
        print("[main] e.2) Load the trained models")
        model = load_trained_models(file_paths, train_data, source_dataset_name, device)
        # Load the val_stats
        print("[main] e.3) Load the val_stats")
        val_stats = load_val_stats(model_input_path, 'val_stats_trained_model.json') 
        # Do Jind
        print("[main] f) Do Jind")
        jind.train(target_data=test_data, model=model, val_stats=val_stats)

    else:
        print('[main] Warning: Trained JIND Multi with this data for the first time')
        jind.train(target_data = test_data)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JindMulti')
    parser.add_argument('-dt', '--DATA_TYPE', type=str, required=True, help='Dataset name') 
    parser.add_argument('-s', '--SOURCE_DATASET_NAME', type=str, help='Name or ID of source dataset') 
    parser.add_argument('-t', '--TARGET_DATASET_NAME', type=str, required=True, help='Name or ID of target dataset') 
    parser.add_argument('-p', '--PATH_WD', type=str, required=True, help='Path to jind_multi folder') 
    args = parser.parse_args()
    main(args)

# pancreas:  python Main.py -dt pancreas -s 0 -t 2 -p /home/jsanchoz/data/josebas/JIND_Iterative/JIND-continual_integration/jind_multi  
 