from JindWrapper import JindWrapper
from DataLoader import load_data
import argparse

def main(args): 
    # 1) Load data and normalized    
    data = load_data(data_type = args.DATA_TYPE, black_list= args.BLACK_LIST)

    # 2) Train JindWrapper
    train_data = data[data['batch'] != args.TARGET_DATASET_NAME]
    test_data = data[data['batch'] == args.TARGET_DATASET_NAME]
    # test_data = test_data.copy() ###
    # test_data.loc[:, 'labels'] = '' ###

    jind = JindWrapper(train_data=train_data, source_dataset_name=args.SOURCE_DATASET_NAME, output_path = args.PATH_WD+'/output/'+ args.DATA_TYPE)
    jind.train(test_data)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JindMulti')
    parser.add_argument('-dt', '--DATA_TYPE', type=str, required=True, help='Dataset name') 
    parser.add_argument('-s', '--SOURCE_DATASET_NAME', type=str, help='Name or ID of source dataset') 
    parser.add_argument('-t', '--TARGET_DATASET_NAME', type=str, required=True, help='Name or ID of target dataset') 
    parser.add_argument('-p', '--PATH_WD', type=str, required=True, help='Path to jind_multi folder') 
    parser.add_argument('-bl', '--BLACK_LIST', nargs='+', help='List with the sample labels to be removed from your dataset')
    args = parser.parse_args()
    main(args)

# pancreas:  python Main.py -dt pancreas -s 0 -t 2 -p /home/jsanchoz/data/josebas/JIND_Iterative/JIND-continual_integration/jind_multi -bl "dropped"
# breast:    python Main.py -dt breast -s CID4535 -t CID44971 -p /home/jsanchoz/data/josebas/JIND_Iterative/JIND-continual_integration/jind_multi