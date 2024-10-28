

import argparse
import os
import timeit
import datetime
import pandas as pd
import torch
import ast
from jind_multi.jind_wrapper import run_single_mode, run_multi_mode, run_combined_mode
from jind_multi.data_loader import load_and_process_data
from jind_multi.config_loader import get_config
from jind_multi.utils import load_config_from_file

def remove_pth_files(folder_path):
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.pth'):
                file_path = os.path.join(root, filename)
                os.remove(file_path)

def run_comparison(args, data, trial):
    """Run the comparison between JIND Single, Multi, and Combine modes."""
    columns = ['num_train_batches', 'rej%', 'raw_acc%', 'eff_acc%', 'mAP%']
    columns_time = ['time']
    jind_multi_results = pd.DataFrame(columns=columns)
    jind_combine_results = pd.DataFrame(columns=columns)
    time_jind_multi = pd.DataFrame(columns=columns_time)
    time_jind_combine = pd.DataFrame(columns=columns_time)
    path_out = args.OUTPUT_PATH

    ### 1) Run Jind Single Mode
    start_time = timeit.default_timer()
    _, raw_acc_per, eff_acc_per, mAP_per, rejected_per = run_single_mode(
                                                    data=data, 
                                                    source_dataset_name=args.SOURCE_DATASET_NAME, 
                                                    target_dataset_name=args.TARGET_DATASET_NAME, 
                                                    path=path_out + f'/Single-trial_{trial}')
                        
    torch.cuda.empty_cache() # free space in GPU
    duration = datetime.timedelta(seconds=timeit.default_timer() - start_time)

    # write results
    run_jind_simple = {
                        'num_train_batches': 1,
                        'rej%': rejected_per,
                        'raw_acc%': raw_acc_per,
                        'eff_acc%': eff_acc_per,
                        'mAP%': mAP_per
                        }

    run_time_jind_simple = {
                        'num_train_batches': 1,
                        'time': str(duration)
                        }

    jind_multi_results = jind_multi_results.append(run_jind_simple, ignore_index=True)
    time_jind_multi = time_jind_multi.append(run_time_jind_simple, ignore_index=True)
    remove_pth_files(path_out + f'/Single-trial_{trial}')  
    print('[Run Comparison] Run Jind Single Mode ... -> DONE')

    ### 2) Run Multi and Combine iteratively adding another batch
    intermediate_batches = data[~data['batch'].isin([args.SOURCE_DATASET_NAME, args.TARGET_DATASET_NAME])].batch.unique().tolist()
    intermediate_batches.sort()
    train_dataset_names = [args.SOURCE_DATASET_NAME] 
   
    for i, batch in enumerate(intermediate_batches):
   
        ### a) Run Jind Multi Mode
        train_dataset_names.append(batch)
        print('[Run Comparison][Jind Multi]', train_dataset_names)
        
        start_time = timeit.default_timer()
        _, raw_acc_per, eff_acc_per, mAP_per, rejected_per = run_multi_mode(
                                                        data=data, 
                                                        train_dataset_names=train_dataset_names, 
                                                        target_dataset_name=args.TARGET_DATASET_NAME, 
                                                        path=path_out + f'/Multi-trial_{trial}/' + f'train_inter_{i+1}_set', 
                                                        source_dataset_name=args.SOURCE_DATASET_NAME)
        duration = datetime.timedelta(seconds=timeit.default_timer() - start_time)
        torch.cuda.empty_cache() # free space in GPU

        run_jind_multi = {
                       'num_train_batches': len(train_dataset_names),
                       'rej%': rejected_per,
                       'raw_acc%': raw_acc_per,
                       'eff_acc%': eff_acc_per,
                       'mAP%': mAP_per
                       }

        run_time_jind_multi = {
                               'num_train_batches': len(train_dataset_names),
                               'time': str(duration)
                               }
        jind_multi_results = jind_multi_results.append(run_jind_multi, ignore_index=True)
        time_jind_multi = time_jind_multi.append(run_time_jind_multi, ignore_index=True)
        remove_pth_files(path_out + f'/Multi-trial_{trial}/' + f'train_inter_{i+1}_set')

        ### b) Run Jind Combine Mode
        print('[Run Comparison][Jind Combine]', train_dataset_names)
        start_time = timeit.default_timer()
        _, raw_acc_per, eff_acc_per, mAP_per, rejected_per = run_combined_mode(
                                                       data=data, 
                                                   source_dataset_names=train_dataset_names, 
                                                        target_dataset_name=args.TARGET_DATASET_NAME, 
                                                       path=path_out + f'/Combine-trial_{trial}/' + f'train_inter_{i+1}_set')
        torch.cuda.empty_cache() # free space in gpu
        duration = datetime.timedelta(seconds=timeit.default_timer() - start_time)
       
        run_jind_combine = {
                           'num_train_batches': len(train_dataset_names),
                            'rej%': rejected_per,
                           'raw_acc%': raw_acc_per,
                           'eff_acc%': eff_acc_per,
                           'mAP%': mAP_per
                           }   
        run_time_jind_combine = {
                            'num_train_batches': len(train_dataset_names),
                            'time': str(duration)
                           }
     
        jind_combine_results = jind_combine_results.append(run_jind_combine, ignore_index=True)
        time_jind_combine = time_jind_combine.append(run_time_jind_combine, ignore_index=True)
        remove_pth_files(path_out + f'/Combine-trial_{trial}/' + f'train_inter_{i+1}_set')

    # Combine and save results
    run_results = pd.merge(jind_multi_results, jind_combine_results, on='num_train_batches', how='outer', suffixes=(' - Jind Multi', ' - Jind Combine'))
    run_time = pd.merge(time_jind_multi, time_jind_combine, on='num_train_batches', how='outer', suffixes=(' - Jind Multi', ' - Jind Combine'))

    run_results.to_excel(path_out + f'/run_results_{trial}.xlsx', index=False)
    run_results.set_index('num_train_batches', inplace=True)
    run_time.to_excel(path_out + f'/time_execution_run_{trial}.xlsx', index=False)
    run_time.set_index('num_train_batches', inplace=True)
    print(run_results)
    print(run_time)
  
def parse_args():    
    parser = argparse.ArgumentParser(description='Compare JIND Methods')
    parser.add_argument('--config', type=str, help='Path to JSON config file')
    parser.add_argument('--PATH', type=str, help='Path to the ann object dataset file with gene expression data of shape cells_id x genes')
    parser.add_argument('--BATCH_COL', type=str, help='Name of the batch column')
    parser.add_argument('--LABELS_COL', type=str, help='Name of the labels column')
    parser.add_argument('--SOURCE_DATASET_NAME', type=str, default=None, help='Name or ID of source dataset') 
    parser.add_argument('--TARGET_DATASET_NAME', type=str, help='Name or ID of target dataset') 
    parser.add_argument('--OUTPUT_PATH', type=str, help='Output path to save results and trained model')
    parser.add_argument('--NUM_FEATURES', type=int, default=5000, help='Optional. Number of genes to consider for modeling, default is 5000')
    parser.add_argument('--MIN_CELL_TYPE_POPULATION', type=int, default=100, help='Optional. For each batch, the minimum number of cells per cell type necessary for modeling. If this requirement is not met in any batch, the samples belonging to this cell type are removed from all batches')
    parser.add_argument('--N_TRIAL', type=int, default=0, help='Number of the trial experiment')
    parser.add_argument('--USE_GPU', type=ast.literal_eval, default=True, help='Optional. Use CUDA if available (True/False), default is True')
    return parser.parse_args()

def main():
    args = parse_args()

    if args.config:
        config_data = load_config_from_file(args.config)
        # Override arguments with values from config file if they are not None
        for key, value in config_data.items():
            if value is not None:
                setattr(args, key, value)

    config = get_config()
    config['data']['num_features'] = args.NUM_FEATURES
    config['data']['min_cell_type_population'] = args.MIN_CELL_TYPE_POPULATION
    config['train_classifier']['cuda'] = args.USE_GPU  
    config['GAN']['cuda'] = args.USE_GPU  
    config['ftune']['cuda'] = args.USE_GPU  

    # Load Data and Normalize
    data = load_and_process_data(args, config) 
    print(f'USE_GPU: {args.USE_GPU}')
    print(f'Arguments: {args}')

    # Execute trial
    run_comparison(args, data, args.N_TRIAL) 

if __name__ == "__main__":
    main()

