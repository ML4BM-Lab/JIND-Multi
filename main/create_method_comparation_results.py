from JindWrapper import JindWrapper, run_single_mode, run_multi_mode, run_combined_mode
from DataLoader import load_data
import argparse
import pandas as pd
import numpy as np
import os
import timeit
import datetime
import torch

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

    ### Run Jind Single Mode
    start_time = timeit.default_timer()
    _, raw_acc_per, eff_acc_per, mAP_per, rejected_per = run_single_mode(data=data, source_dataset_name=args.SOURCE_DATASET_NAME, target_dataset_name=args.TARGET_DATASET_NAME, 
                                        path=args.PATH_WD + '/output/'+ args.DATA_TYPE + f'/Single-trial_{trial}')
    torch.cuda.empty_cache() # free space in gpu
    duration = datetime.timedelta(seconds=timeit.default_timer() - start_time)

    run_jind_simple = {'num_train_batches': 1,
                       'rej%': rejected_per,
                       'raw_acc%': raw_acc_per,
                       'eff_acc%': eff_acc_per,
                       'mAP%': mAP_per}
    run_time_jind_simple = {'num_train_batches': 1,
                        'time': str(duration)}

    jind_multi_results = jind_multi_results.append(run_jind_simple, ignore_index=True)
    time_jind_multi = time_jind_multi.append(run_time_jind_simple, ignore_index=True)

    # remove actual runs' .pth 
    remove_pth_files(args.PATH_WD + '/output/'+ args.DATA_TYPE + f'/Single-trial_{trial}')  
    print('[create_method_comparation_results] Run Jind Single Mode ... -> DONE')

    ### Run Multi and Combine iteratively adding another batch
    intermediate_batches = data[~data['batch'].isin([args.SOURCE_DATASET_NAME, args.TARGET_DATASET_NAME])].batch.unique().tolist()
    intermediate_batches.sort()
    train_dataset_names = [args.SOURCE_DATASET_NAME] 
   
    for i in range(0,len(intermediate_batches)):
        ### Jind Multi
        train_dataset_names.append(intermediate_batches[i])
        print('[create_method_comparation_results][Jind Multi]', train_dataset_names)
        
        start_time = timeit.default_timer()
        _, raw_acc_per, eff_acc_per, mAP_per, rejected_per = run_multi_mode(data=data, train_dataset_names=train_dataset_names, target_dataset_name=args.TARGET_DATASET_NAME, 
                                    path=args.PATH_WD + '/output/'+ args.DATA_TYPE + f'/Multi-trial_{trial}/' + f'train_inter_{i+1}_set', source_dataset_name=args.SOURCE_DATASET_NAME)
        duration = datetime.timedelta(seconds=timeit.default_timer() - start_time)
        torch.cuda.empty_cache() # free space in gpu

        run_jind_multi = {'num_train_batches': len(train_dataset_names),
                          'rej%': rejected_per,
                          'raw_acc%': raw_acc_per,
                          'eff_acc%': eff_acc_per,
                          'mAP%': mAP_per}
        run_time_jind_multi = {'num_train_batches': len(train_dataset_names),
                           'time': str(duration)}
        jind_multi_results = jind_multi_results.append(run_jind_multi, ignore_index=True)
        time_jind_multi = time_jind_multi.append(run_time_jind_multi, ignore_index=True)

        # remove this runs' .pth 
        remove_pth_files(args.PATH_WD + '/output/'+ args.DATA_TYPE + f'/Multi-trial_{trial}/' + f'train_inter_{i+1}_set')

        ### Jind Combine
        print('[create_method_comparation_results][Jind Combine]', train_dataset_names)
        start_time = timeit.default_timer()
        _, raw_acc_per, eff_acc_per, mAP_per, rejected_per = run_combined_mode(data=data, source_dataset_names=train_dataset_names, target_dataset_name=args.TARGET_DATASET_NAME, 
                       path=args.PATH_WD + '/output/'+ args.DATA_TYPE + f'/Combine-trial_{trial}/' + f'train_inter_{i+1}_set')
        torch.cuda.empty_cache() # free space in gpu
        duration = datetime.timedelta(seconds=timeit.default_timer() - start_time)
       
        run_jind_combine = {'num_train_batches': len(train_dataset_names),
                            'rej%': rejected_per,
                            'raw_acc%': raw_acc_per,
                            'eff_acc%': eff_acc_per,
                            'mAP%': mAP_per}   
        run_time_jind_combine = {'num_train_batches': len(train_dataset_names),
                           'time': str(duration)}
     
        jind_combine_results = jind_combine_results.append(run_jind_combine, ignore_index=True)
        time_jind_combine = time_jind_combine.append(run_time_jind_combine, ignore_index=True)

        # remove this runs' .pth 
        remove_pth_files(args.PATH_WD + '/output/'+ args.DATA_TYPE + f'/Combine-trial_{trial}/' + f'train_inter_{i+1}_set')

    run_results = pd.merge(jind_multi_results, jind_combine_results, on='num_train_batches', how='outer', suffixes=(' - Jind Multi', ' - Jind Combine'))
    print(run_results)
    run_time = pd.merge(time_jind_multi, time_jind_combine, on='num_train_batches', how='outer', suffixes=(' - Jind Multi', ' - Jind Combine'))
    print(run_time)

    run_results.to_excel(args.PATH_WD + '/output/'+ args.DATA_TYPE + f'/run_results_{trial}.xlsx', index=False)
    run_results.set_index('num_train_batches', inplace=True)
    run_time.to_excel(args.PATH_WD + '/output/'+ args.DATA_TYPE + f'/time_execution_run_{trial}.xlsx', index=False)
    run_time.set_index('num_train_batches', inplace=True)
    return run_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JindMulti')
    parser.add_argument('-dt', '--DATA_TYPE', type=str, required=True, help='Dataset name') 
    parser.add_argument('-s', '--SOURCE_DATASET_NAME', type=str, help='Name or ID of source dataset') 
    parser.add_argument('-t', '--TARGET_DATASET_NAME', type=str, required=True, help='Name or ID of target dataset') 
    parser.add_argument('-p', '--PATH_WD', type=str, required=True, help='Path to jind_multi folder') 
    parser.add_argument('-bl', '--BLACK_LIST', nargs='+', help='List with the sample labels to be removed from your dataset')
    parser.add_argument('-nt', '--N_TRIAL', type=int, required=True, help='Number of the trial experiment')
    args = parser.parse_args()

    ### Load Data and Normalize
    data = load_data(data_type = args.DATA_TYPE, source_dataset_name=args.SOURCE_DATASET_NAME) #  caso 2)
    # caso 3) data = load_data(data_type = args.DATA_TYPE, black_list= args.BLACK_LIST, source_dataset_name=args.SOURCE_DATASET_NAME, preserve_target_labels_dataset_name=args.TARGET_DATASET_NAME)
    run_results = main(args, data, args.N_TRIAL) # Results of the trial
    
# Test Dataset
# pancreas:    python create_method_comparation_results.py -dt pancreas -s 0 -t 3 -p /home/jsanchoz/data/josebas/JIND_Iterative/JIND-continual_integration/jind_multi -bl "dropped" "endothelial"
# human_brain: python create_method_comparation_results.py -dt brain_neurips -s C4 -t C7 -p /home/jsanchoz/data/josebas/JIND_Iterative/JIND-continual_integration/jind_multi -nt
# nsclc:       python create_method_comparation_results.py -dt nsclc_lung -s Donor5 -t Donor2  -p /home/jsanchoz/data/josebas/JIND_Iterative/JIND-continual_integration/jind_multi -nt
# bmmc_atac:   python create_method_comparation_results.py -dt bmmc_atac -s s4d8 -t s3d3 -p /home/jsanchoz/data/josebas/JIND_Iterative/JIND-continual_integration/jind_multi -nt

# atac2: nohup python 
    # heart: nohup python create_method_comparation_results.py -dt heart_atac2 -s heart_sample_39 -t heart_sample_14 -p /home/jsanchoz/data/josebas/JIND_Iterative/JIND-continual_integration/jind_multi -nt 0 > /home/jsanchoz/data/josebas/JIND_Iterative/JIND-continual_integration/jind_multi/main/logs/heart_atac2_try0.log &
    # cerebrum: nohup python create_method_comparation_results.py -dt cerebrum_atac2 -s cerebrum_sample_6 -t cerebrum_sample_66 -p /home/jsanchoz/data/josebas/JIND_Iterative/JIND-continual_integration/jind_multi -nt 0 > /home/jsanchoz/data/josebas/JIND_Iterative/JIND-continual_integration/jind_multi/main/logs/cerebrum_atac2_try0.log &
    # kidney: python create_method_comparation_results.py -dt kidney_atac2 -s kidney_sample_3 -t kidney_sample_67 -p /home/jsanchoz/data/josebas/JIND_Iterative/JIND-continual_integration/jind_multi -nt 0
    # liver: python create_method_comparation_results.py -dt liver_atac2 -s liver_sample_35 -t liver_sample_9 -p /home/jsanchoz/data/josebas/JIND_Iterative/JIND-continual_integration/jind_multi -nt 0
    
# Extra Dataset
# breast:      python create_method_comparation_results.py -dt breast -s CID4495 -t CID3838 -p /home/jsanchoz/data/josebas/JIND_Iterative/JIND-continual_integration/jind_multi -nt
# test:        python create_method_comparation_results.py -dt test -s Source -t Target -p /home/jsanchoz/data/josebas/JIND_Iterative/JIND-continual_integration/jind_multi -nt

 