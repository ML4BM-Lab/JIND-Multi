import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset
from Utils import dimension_reduction, preprocess, filter_cells, create_scanpy_embeddings, create_scanpy_umap, create_umap_from_dataframe, find_saved_models, load_trained_models, load_val_stats
from JindWrapper import JindWrapper
from JindLib import JindLib
from DataLoader import load_data
import argparse
from ConfigLoader import get_config
import gc
import os
from glob import glob
import ast

def heatmap_celltypes(data, path_save, label_name='labels', database_name='v3'):
    batches = data['batch'].unique()
    dfs = []  # Lista para almacenar los DataFrames de cada batch
    for batch in batches:
        batch_data = data[data.batch == batch]
        label_counts = batch_data[label_name].value_counts()
        label_counts
        dfs.append(label_counts)  # Agregar el DataFrame de este batch a la lista
    df = pd.concat(dfs, axis=1)  # Concatenar todos los DataFrames en uno solo
    df.columns = batches
    
    n_rows, n_cols = df.shape
    figsize = (n_cols * 3, n_rows * 1.5)  # Puedes ajustar los factores de escala según sea necesario

    plt.figure(figsize=figsize)
    sns.heatmap(df.T, cmap="YlGnBu", annot=True, fmt="d", cbar=True)
    plt.yticks(rotation=0)
    plt.xticks(rotation=33)
    plt.title(f'Counts of Labels per Batch in {database_name}', fontsize=16)
    plt.xlabel(label_name, fontsize=14)
    plt.ylabel('Batch', fontsize=14)
    plt.savefig(path_save+f'/{database_name}_heatmap_label_count_batches.png')
    plt.close()

def make_label_names_consistent(data, labels_col):
    if labels_col == 'JIND_Res_1': # args.LABELS_COL:
        data['labels'] = data['labels'].replace('CD4', 'CD4 cells')
        data['labels'] = data['labels'].replace('CD8', 'CD8 cells')
    return data

def load_scamara_data(args):
    path = args.PATH
    path_save = args.OUTPUT_PATH + f'/{args.LABELS_COL}'
    os.makedirs(path_save, exist_ok=True)
   
    # V3 - Tiene menos muestras y está anotado
    ##############################################
    adata_v3 = sc.read(path+'/Python_scVI_adata_V3_state4_Res.h5ad')

    # Ploteamos el dataset entero:
    #sc.pl.scatter(adata_v3, basis='umap', color=['manual_celltype_annotation_high', 'Product_norm'], frameon=False, show=False)
    #plt.savefig(f'{path_save}/umap_v3_annotated_batch_labels.png', bbox_inches="tight")
    select_batches = ast.literal_eval(args.TRAIN_DATASETS_NAMES)
    adata_v3 = adata_v3[adata_v3.obs['Product_norm'].isin(select_batches)]
   
    # Ploteamos el dataset filtrado de entrenamiento:
    #sc.pl.scatter(adata_v3, basis='umap', color=['manual_celltype_annotation_high', 'Product_norm'], frameon=False, show=False)
    #plt.savefig(f'{path_save}/umap_v3_txiki_annotated_batch_labels.png', bbox_inches="tight")
    data_v3 = adata_v3.to_df()
    data_v3['batch'] = adata_v3.obs['Product_norm']
    data_v3['labels'] = adata_v3.obs[args.LABELS_COL]
    
    # Filtrar las filas donde 'labels' no sea igual a 'Ribosomal enriched'. El dataset anotado queremos saber todo.  
    data_v3 = data_v3[data_v3['labels'] != 'Unknown']
    
    # Common genes and labels in annotated batches
    batches_v3 = data_v3['batch'].unique()
    common_genes_v3 = list(set.intersection(*[set(data_v3[data_v3['batch'] == batch].columns) for batch in batches_v3]))
    common_genes_v3.sort()
    data_v3 = data_v3[list(common_genes_v3)]
    common_labels = list(set.intersection(*[set(data_v3[data_v3['batch'] == batch]['labels']) for batch in batches_v3]))
    common_labels.sort()
    data_v3 = data_v3[data_v3['labels'].isin(common_labels)]
    # Heatmap of labels counts per batch
    heatmap_celltypes(data_v3, path_save, label_name='labels', database_name = f'v3_{args.LABELS_COL}')
    
    # V4 - Tiene más muestras y es el que queremos probar como sale la anotación
    ###################################################################################
    adata_v4 = sc.read(path+'/Python_scVI_adata_V4_state4_Res.h5ad')
    data_v4 = adata_v4.to_df()
    data_v4['batch'] = adata_v4.obs['Product_norm']

    if not os.path.exists(f'{path_save}/results'):
        os.makedirs(f'{path_save}/results')
    
    #if args.LABELS_COL in adata_v4.obs:
    #    print('There is a ground-truth available for the target batches we want to predict\n')
    #    data_v4['ground_truth'] = adata_v4.obs[args.LABELS_COL]
        # Heatmap of labels counts per batch
        #heatmap_celltypes(data_v4, path_save, label_name='ground_truth', database_name = f'v4_{args.LABELS_COL}')
        
    #    if not os.path.exists(f'{path_save}/results'):
    #    os.makedirs(f'{path_save}/results')
        #data_v4.to_csv(f'{path_save}/results/data_with_ground_truth.csv')
        #print(f'Ground Truth of the samples we want to predict saved in {path_save}/results/')
        #data_v4.drop(columns=['ground_truth'], inplace=True)
    
    data_v4['labels'] = adata_v4.obs[args.LABELS_COL] 
    data_v4['labels'] = data_v4['labels'].replace('Unknown', 'Unassigned')
    data_v4 = make_label_names_consistent(data=data_v4, labels_col=args.LABELS_COL)
    batches_v4 = data_v4['batch'].unique()
    
    # Common genes in no-annotated batches
    common_genes_v4 = list(set.intersection(*[set(data_v4[data_v4['batch'] == batch].columns) for batch in batches_v4]))
    common_genes_v4.sort()
    data_v4 = data_v4[list(common_genes_v4)]
    
    # Get the intersection of common genes between v3 and v4
    common_genes_intersect = set(common_genes_v3).intersection(common_genes_v4)
    # Filter v3 and v4 data to include only the common genes
    data_v3 = data_v3[list(common_genes_intersect)]
    data_v4 = data_v4[list(common_genes_intersect)]
    # Reorder columns to move 'batch' and 'labels' to the end
    data_v3 = data_v3.reindex(columns=[col for col in data_v3.columns if col not in ['batch', 'labels']] + ['batch', 'labels'])
    data_v4 = data_v4.reindex(columns=[col for col in data_v4.columns if col not in ['batch', 'labels']] + ['batch', 'labels'])
    
    # Apply the lambda function to add "target_" before each name in the "batch" column
    data_v4['batch'] = data_v4['batch'].apply(lambda x: 'target_' + x)
    
    del adata_v3, adata_v4
    gc.collect()
    return data_v3, data_v4


def main(args): 
    # 0) Setting the training configuration (you can modify more things here)
    config = get_config()
    config['data']['num_features'] = args.NUM_FEATURES
    config['data']['min_cell_type_population'] = args.MIN_CELL_TYPE_POPULATION

    config = config['data']
    use_cuda = get_config()['train_classifier']['cuda']
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 1) Read Raw Data
    data_v3, data_v4 = load_scamara_data(args)
    print("[main] 1) Read Raw Data")
   
    # 2) Train JIND Multi for each Target
    print("[main] 2) Train JIND Multi for each Target")
    print(args.LIST_TARGET_DATASET_NAMES)
    list_target_batches = ast.literal_eval(args.LIST_TARGET_DATASET_NAMES)
    
    for target_batch in list_target_batches:
        # a) Select target batch
        target_batch = f'target_{target_batch}'
        target_data_raw = data_v4[data_v4.batch == target_batch]
        print(f"[main] a) Selected target batch {target_data_raw}")
        
        # b) Add annotated data & non-annotated in the same data
        print("[main] b) Add annotated data & non-annotated in the same data")
        data = pd.concat([data_v3, target_data_raw], axis=0)
        
        # c) Processing data
        print("[main] c) Processing data")
        data = preprocess(data, count_normalize=config['count_normalize'], log_transformation=config['log_transformation'])
        data = dimension_reduction(data, num_features=config['num_features'])
        
        # In this case we just wanna filter train sample cells
        data_train = filter_cells(data[~(data.batch == target_batch)], min_cell_type_population=args.MIN_CELL_TYPE_POPULATION)  
        data = pd.concat([data_train, data[data.batch == target_batch]])
        data = data.reindex(sorted(data.columns), axis=1)  # Reorder columns
        
        # d) Initialize JindWrapper object
        print("[main] d) Initialize JindWrapper object")
        train_data = data[data['batch'] != target_batch]
        test_data = data[data['batch'] == target_batch] #.drop(columns=['labels'])

        source_dataset_name = args.SOURCE_DATASET_NAME 
        output_path = f'{args.OUTPUT_PATH}/{args.LABELS_COL}/predictions/{target_batch}'
        jind = JindWrapper(
                        train_data=train_data, 
                        train_dataset_names=ast.literal_eval(args.TRAIN_DATASETS_NAMES),  
                        source_dataset_name=source_dataset_name, 
                        output_path=output_path
                    )

        # e) Train the JindWrapper   
        # Check if there is already a trained model available
        model_input_path = os.path.abspath(os.path.join(output_path, '../../results'))
        file_paths = find_saved_models(model_input_path, train_data)

        if file_paths:
            print('[main] Warning: Trained Models detected')
            print(f'[main] File Paths: {file_paths}')
            # Load the trained models
            print("[main] e.2) Load the trained models")
            model = load_trained_models(file_paths, train_data, args.SOURCE_DATASET_NAME)
            # Load the val_stats
            print("[main] e.3) Load the val_stats")
            val_stats = load_val_stats(model_input_path, 'val_stats_trained_model.json') 
            # Do Jind
            print("[main] f) Do Jind")
            jind.train(target_data=test_data, model=model, val_stats=val_stats)

        else:
            print('[main] Warning: Trained JIND Multi with this data for the first time')
            jind.train(target_data = test_data)

        gc.collect()
        torch.cuda.empty_cache()
        del train_data, test_data, data, jind

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main script to execute JindMulti and annotate a target batch using several annotated batches')

    parser.add_argument('--PATH', type=str, default='/scratch/jsanchoz/JIND-Multi/resources/data/scamara/data', help='Path to the scamara data where we have the h5ad files')
    parser.add_argument('--OUTPUT_PATH', type=str, default='/scratch/jsanchoz/JIND-Multi/resources/data/scamara', help='Output path to save results and trained model')
    parser.add_argument('--LABELS_COL', type=str, default='JIND_Res_1', help='Name of the labels column (options: JIND_Res_1, JIND_Res_2, JIND_Res_3, JIND_Res_4)')
    parser.add_argument('--SOURCE_DATASET_NAME', type=str, default='Goo_Pt_245', help='Name or ID of the source batch') 
    parser.add_argument('--TRAIN_DATASETS_NAMES', type=str, default="['Goo_Pt_245', 'Goo_Pt_253', 'Goo_Pt_110', 'Goo_Pt_116', 'Goo_Pt_125', 'Goo_Pt_263', 'Goo_Pt_276']", help='List of training batch names in the desired order, starting with the source batch, followed by intermediate batches in the order they should be processed')
    parser.add_argument('--LIST_TARGET_DATASET_NAMES', type=str, help='List of target batches')
    parser.add_argument('--NUM_FEATURES', type=int, default=5000, help='Optional. Number of genes to consider for modeling, default is 5000')
    parser.add_argument('--MIN_CELL_TYPE_POPULATION', type=int, default=100, help='Optional. For each batch, the minimum number of cells per cell type necessary for modeling. If this requirement is not met in any batch, the samples belonging to this cell type are removed from all batches')

    args = parser.parse_args()
    main(args)