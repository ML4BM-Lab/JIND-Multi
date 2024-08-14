from jind_multi.jind_wrapper import run_single_mode, run_multi_mode, run_combined_mode
from jind_multi.data_loader import load_and_process_data
from jind_multi.config_loader import get_config
import argparse
import pandas as pd
import numpy as np
import os
import timeit
import datetime
import torch
import ast

def remove_pth_files(folder_path):
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.pth'):
                file_path = os.path.join(root, filename)
                os.remove(file_path)

def main(args, data, trial): 
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
                        
    torch.cuda.empty_cache() # free space in gpu
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

    # remove actual runs' .pth (if you want to save space)
    remove_pth_files(path_out + f'/Single-trial_{trial}')  
    print('[create_method_comparation_results] Run Jind Single Mode ... -> DONE')

    ### 2) Run Multi and Combine iteratively adding another batch
    intermediate_batches = data[~data['batch'].isin([args.SOURCE_DATASET_NAME, args.TARGET_DATASET_NAME])].batch.unique().tolist()
    intermediate_batches.sort()
    train_dataset_names = [args.SOURCE_DATASET_NAME] 
   
    for i in range(0,len(intermediate_batches)):
        ### a) Run Jind Multi Mode
        train_dataset_names.append(intermediate_batches[i])
        print('[create_method_comparation_results][Jind Multi]', train_dataset_names)
        
        start_time = timeit.default_timer()
        _, raw_acc_per, eff_acc_per, mAP_per, rejected_per = run_multi_mode(
                                                        data=data, 
                                                        train_dataset_names=train_dataset_names, 
                                                        target_dataset_name=args.TARGET_DATASET_NAME, 
                                                        path=path_out + f'/Multi-trial_{trial}/' + f'train_inter_{i+1}_set', 
                                                        source_dataset_name=args.SOURCE_DATASET_NAME)
        duration = datetime.timedelta(seconds=timeit.default_timer() - start_time)
        torch.cuda.empty_cache() # free space in gpu

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

        # remove this runs' .pth 
        remove_pth_files(path_out + f'/Multi-trial_{trial}/' + f'train_inter_{i+1}_set')

        ### b) Run Jind Combine Mode
        print('[create_method_comparation_results][Jind Combine]', train_dataset_names)
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

        # remove this runs' .pth 
        remove_pth_files(path_out + f'/Combine-trial_{trial}/' + f'train_inter_{i+1}_set')

    run_results = pd.merge(jind_multi_results, jind_combine_results, on='num_train_batches', how='outer', suffixes=(' - Jind Multi', ' - Jind Combine'))
    print(run_results)
    run_time = pd.merge(time_jind_multi, time_jind_combine, on='num_train_batches', how='outer', suffixes=(' - Jind Multi', ' - Jind Combine'))
    print(run_time)

    run_results.to_excel(path_out + f'/run_results_{trial}.xlsx', index=False)
    run_results.set_index('num_train_batches', inplace=True)
    run_time.to_excel(path_out + f'/time_execution_run_{trial}.xlsx', index=False)
    run_time.set_index('num_train_batches', inplace=True)
    return run_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JindMulti')
    parser.add_argument('--PATH', type=str, required=True, help='Path to the ann object dataset file with gene expression data of shape cells_id x genes')
    parser.add_argument('--BATCH_COL', type=str, required=True, help='Name of the batch column')
    parser.add_argument('--LABELS_COL', type=str, required=True, help='Name of the labels column')
    parser.add_argument('--SOURCE_DATASET_NAME', type=str, help='Name or ID of source dataset') 
    parser.add_argument('--TARGET_DATASET_NAME', type=str, required=True, help='Name or ID of target dataset') 
    parser.add_argument('--OUTPUT_PATH', type=str, required=True, help='Output path to save results and trained model')
    parser.add_argument('--NUM_FEATURES', type=int, default=5000, help='Optional. Number of genes to consider for modeling, default is 5000')
    parser.add_argument('--MIN_CELL_TYPE_POPULATION', type=int, default=100, help='Optional. For each batch, the minimum number of cells per cell type necessary for modeling. If this requirement is not met in any batch, the samples belonging to this cell type are removed from all batches')
    parser.add_argument('--N_TRIAL', type=int, required=True, help='Number of the trial experiment')
    parser.add_argument('--USE_GPU', type=ast.literal_eval, default=True, help='Optional. Use CUDA if available (True/False), default is True')
    args = parser.parse_args()

    # Setting the training configuration (you can modify more things here)
    config = get_config()
    config['data']['num_features'] = args.NUM_FEATURES
    config['data']['min_cell_type_population'] = args.MIN_CELL_TYPE_POPULATION
    config['train_classifier']['cuda'] = args.USE_GPU  
    config['GAN']['cuda'] = args.USE_GPU  
    config['ftune']['cuda'] = args.USE_GPU  
    print(f'USE_GPU: {args.USE_GPU}')

    # Load Data and Normalize
    data = load_and_process_data(args.PATH, args.BATCH_COL, args.LABELS_COL, config) 
    # Execute trial
    run_results = main(args, data, args.N_TRIAL) 

 