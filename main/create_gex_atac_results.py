from JindWrapper import JindWrapper, run_single_mode, run_multi_mode, run_combined_mode
from DataLoader import load_data
import argparse
import pandas as pd
import numpy as np
import os
import timeit
import datetime

# coger el GEX de un batchX y anotar el ATAC de un batch X
# coger el GEX de un batchX y anotar el ATAC de un batch Y

def main(args, data, trial): 
    ### Run Jind Single Mode
    start_time = timeit.default_timer()
    _, raw_acc_per, eff_acc_per, mAP_per, rejected_per = run_single_mode(data=data, source_dataset_name=args.SOURCE_DATASET_NAME, target_dataset_name=args.TARGET_DATASET_NAME, 
                                        path=args.PATH_WD + '/output/'+ args.DATA_TYPE + f'_{args.SOURCE_DATASET_NAME}_to_{args.TARGET_DATASET_NAME}' + f'/Single-trial_{trial}')
    duration = datetime.timedelta(seconds=timeit.default_timer() - start_time)

    run_jind_simple = {'num_train_batches': 1,
                       'rej%': rejected_per,
                       'raw_acc%': raw_acc_per,
                       'eff_acc%': eff_acc_per,
                       'mAP%': mAP_per}
    run_time_jind_simple = {'num_train_batches': 1,'time': str(duration)}

    jind_results = pd.DataFrame([run_jind_simple])
    time_jind =  pd.DataFrame([run_time_jind_simple])

    jind_results.to_excel(args.PATH_WD + '/output/'+ args.DATA_TYPE + f'_{args.SOURCE_DATASET_NAME}_to_{args.TARGET_DATASET_NAME}' + f'/run_results_{trial}.xlsx', index=False)
    time_jind.to_excel(args.PATH_WD + '/output/'+ args.DATA_TYPE + f'_{args.SOURCE_DATASET_NAME}_to_{args.TARGET_DATASET_NAME}' + f'/time_execution_run_{trial}.xlsx', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Jind in Omics')
    parser.add_argument('-dt', '--DATA_TYPE', type=str, required=True, help='Dataset name') 
    parser.add_argument('-s', '--SOURCE_DATASET_NAME', type=str, help='Name or ID of source dataset') 
    parser.add_argument('-t', '--TARGET_DATASET_NAME', type=str, required=True, help='Name or ID of target dataset') 
    parser.add_argument('-p', '--PATH_WD', type=str, required=True, help='Path to jind_multi folder') 
    parser.add_argument('-nt', '--N_TRIAL', type=int, required=True, help='Number of the trial experiment')
    args = parser.parse_args()

    ### Load Data and Normalize
    data = load_data(data_type = args.DATA_TYPE)
    main(args, data, args.N_TRIAL) # Results of the trial
       
# - coger el GEX de un batchX y anotar el ATAC de un batch X (SINGLE MODE. source = s4d8_gex. target = s4d8_atac)
# bmmc_omics: python  create_gex_atac_results.py -dt bmmc_omics -s s4d8_gex -t s4d8_atac -p /home/jsanchoz/data/josebas/JIND_Iterative/JIND-continual_integration/jind_multi -nt 0

# - coger el GEX de un batchX y anotar el ATAC de un batch Y (SINGLE MODE. source = s4d8_gex. target = s3d3_atac)
# bmmc_omics: python  create_gex_atac_results.py -dt bmmc_omics -s s4d8_gex -t s3d3_atac -p /home/jsanchoz/data/josebas/JIND_Iterative/JIND-continual_integration/jind_multi -nt 0