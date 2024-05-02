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

# # Filtramos solo los batches que scamara me ha dicho
#     select_batches = ['Rod_D10', 'Rod_D10_d7', 'Rod_D14', 'Rod_D14_d7', 'Rod_D18', 'Rod_D18_d7', # son de la "casa" y vienen de donantes sanos
#                     'Goo_Pt_110 ', 'Goo_Pt_116', 'Goo_Pt_125', 'Goo_Pt_245', 'Goo_Pt_253', 'Goo_Pt_263', 'Goo_Pt_276', 'Goo_Pt_282', # vienen de pacientes enfermos, por lo tanto, ese subtipo especial de células que
#                                                                                                                                      # te he comentado anteriormente. Yo creo sinceramente que las muestras de
#                                                                                                                                      # este estudio son de las mejores a la hora de entrenar (Habrá que ver qué
#                                                                                                                                      # resultados te dan).
                    
#                     'Den_Pt_14', 'Den_Pt_15', 'Den_Pt_16', 'Den_Pt_20', 'Den_Pt_21', 'Den_Pt_26', 'Den_Pt_27', 
#                     'Den_Pt_28','Den_Pt_33', 'Den_Pt_34', 'Den_Pt_37', 'Den_Pt_38', 'Den_Pt_40', 'Den_Pt_41', 'Den_Pt_42', 'Den_Pt_43', 'Den_Pt_50', 
#                     'Den_Pt_54', 'Den_Pt_55', 'Den_Pt_56', 'Den_Pt_59', 'Den_Pt_64', 'Lyn_Exp2_Cont', 'Lyn_Exp2_JUN']

def heatmap_celltypes(data, path_save, label_name='labels', database_name='v3'):
    batches = data['batch'].unique()
    dfs = []  # Lista para almacenar los DataFrames de cada batch
    for batch in batches:
        batch_data = data[data.batch == batch]
        label_counts = batch_data[label_name].value_counts()
        dfs.append(label_counts)  # Agregar el DataFrame de este batch a la lista
    df = pd.concat(dfs, axis=1)  # Concatenar todos los DataFrames en uno solo
    plt.figure(figsize=(30, 40))
    sns.heatmap(df.T, cmap="YlGnBu", annot=True, fmt="d", cbar=True)
    plt.yticks(rotation=0)
    plt.xticks(rotation=33)
    plt.title(f'Counts of Labels per Batch in {database_name}', fontsize=16)
    plt.xlabel(label_name, fontsize=14)
    plt.ylabel('Batch', fontsize=14)
    plt.savefig(path_save+f'{database_name}_heatmap_label_count_batches.png')
    plt.close()

def load_scamara_data():
    path = '/home/scamara/data/scamara/JOSEBA'
    path_save = '/home/jsanchoz/data/josebas/JIND_Iterative/JIND-continual_integration/jind_multi/resources/data/scamara/'
    # V3 - Tiene menos muestras y está anotado
    ##############################################
    adata_v3 = sc.read(path+'/Python_scVI_adata_V3_state4.h5ad')
    # Ploteamos el dataset entero:
    sc.pl.scatter(adata_v3, basis='umap', color=['manual_celltype_annotation_high', 'Product_norm'], frameon=False, show=False)
    plt.savefig(f'{path_save}/umap_v3_annotated_batch_labels.png', bbox_inches="tight")
    select_batches = ['Goo_Pt_110', 'Goo_Pt_116', 'Goo_Pt_125', 'Goo_Pt_245', 'Goo_Pt_253', 'Goo_Pt_263', 'Goo_Pt_276'] 
    adata_v3 = adata_v3[adata_v3.obs['Product_norm'].isin(select_batches)]
    # Ploteamos el dataset filtrado de entrenamiento:
    sc.pl.scatter(adata_v3, basis='umap', color=['manual_celltype_annotation_high', 'Product_norm'], frameon=False, show=False)
    plt.savefig(f'{path_save}/umap_v3_txiki_annotated_batch_labels.png', bbox_inches="tight")
    data_v3 = adata_v3.to_df()
    data_v3['batch'] = adata_v3.obs['Product_norm']
    data_v3['labels'] = adata_v3.obs['manual_celltype_annotation_high']
    # Filtrar las filas donde 'labels' no sea igual a 'Ribosomal enriched'  
    data_v3 = data_v3[data_v3['labels'] != 'Ribosomal enriched']
    # Common genes and labels in annotated batches
    batches_v3 = data_v3['batch'].unique()
    common_genes_v3 = list(set.intersection(*[set(data_v3[data_v3['batch'] == batch].columns) for batch in batches_v3]))
    common_genes_v3.sort()
    data_v3 = data_v3[list(common_genes_v3)]
    common_labels = list(set.intersection(*[set(data_v3[data_v3['batch'] == batch]['labels']) for batch in batches_v3]))
    common_labels.sort()
    data_v3 = data_v3[data_v3['labels'].isin(common_labels)]
    # Heatmap of labels counts per batch
    heatmap_celltypes(data_v3, path_save, database_name = 'v3')
    # V4 - Tiene más muestras y es el que queremos probar como sale la anotación
    ###################################################################################
    adata_v4 = sc.read(path+'/Python_scVI_adata_V4_state4.h5ad')
    data_v4 = adata_v4.to_df()
    data_v4['batch'] = adata_v4.obs['Product_norm']
    if 'manual_celltype_annotation_high' in adata_v4.obs:
        print('There is a ground-truth available for the target batches we want to predict\n')
        data_v4['ground_truth'] = adata_v4.obs['manual_celltype_annotation_high']
        # Heatmap of labels counts per batch
        heatmap_celltypes(data_v4, path_save, label_name = 'ground_truth', database_name = 'v4')
        data_v4.to_csv(f'{path_save}/results/data_with_ground_truth.csv')
        print(f'Ground Truth of the samples we want to predict saved in {path_save}/results/')
        data_v4.drop(columns=['ground_truth'], inplace=True)
    data_v4['labels'] = "Unassigned"  
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
    return data_v3, data_v4


def main(args): 
    config={}
    config = get_config(config)['data']
    use_cuda = get_config()['train_classifier']['cuda']
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 1) Read Raw Data
    data_v3, data_v4 = load_scamara_data()
    print("[main] 1) Read Raw Data")
   
    # 2) Train JIND Multi for each Target
    print("[main] 2) Train JIND Multi for each Target")
    for target_batch in args.list_target_batches:
        # local variables
        target_batch = f'target_{target_batch}'
        path = '/home/jsanchoz/data/josebas/JIND_Iterative/JIND-continual_integration/jind_multi/resources/data/scamara/'

        # a) Select target batch
        target_data_raw = data_v4[data_v4.batch == target_batch]
        print(f"[main] a) Selected target batch {target_data_raw}")
        # b) Add annotated data & non-annotated in the same data
        print("[main] b) Add annotated data & non-annotated in the same data")
        data = pd.concat([data_v3, target_data_raw], axis=0)
        # c) Processing data
        print("[main] c) Processing data")
        data = preprocess(data, count_normalize=config['count_normalize'], log_transformation=config['log_transformation'])
        data = dimension_reduction(data, num_features=config['num_features'])
        data = filter_cells(data, min_cell_type_population=5)
        data = data.reindex(sorted(data.columns), axis=1)  # Reorder columns
        
        # d) Initialize JindWrapper object
        print("[main] d) Initialize JindWrapper object")
        train_data = data[data['batch'] != target_batch]
        test_data = data[data['batch'] == target_batch] #.drop(columns=['labels'])

        source_dataset_name = 'Goo_Pt_245' # el batch anotado que más células tiene
        output_path = f'{path}predictions/{target_batch}'
        jind = JindWrapper(train_data=train_data, source_dataset_name=source_dataset_name, output_path = output_path) 

        # e) Train the JindWrapper    
        # Check if there is already a trained model available
        model_input_path = os.path.abspath(os.path.join(output_path, '../../results'))
        file_paths = find_saved_models(model_input_path, train_data)

        if file_paths:
            print('[main] Warning: Trained Models detected')
            print(f'[main] File Paths: {file_paths}')
            # e.2) Load the trained models
            print("[main] e.2) Load the trained models")
            model = load_trained_models(file_paths, train_data, source_dataset_name, device)
            # e.3) Load the val_stats
            print("[main] e.3) Load the val_stats")
            val_stats = load_val_stats(model_input_path, 'val_stats_trained_model.json') 
            # f) Do Jind
            print("[main] f) Do Jind")
            jind.train(target_data=test_data, model=model, val_stats=val_stats)

        else:
            print('[main] Warning: Trained JIND Multi with this data for the first time')
            jind.train(target_data = test_data)


        # # Change target_model name so can't be confussed
        # print("[main] g) Change target_model name so can't be confussed")
        # output_path = output_path+'/'
        # original_path = os.path.join(output_path, f'{target_batch}_bestbr_ftune.pth')
        # modified_path = os.path.join(output_path, f'target_{target_batch}_bestbr_ftune.pth')
        # os.rename(original_path, modified_path)

        # gc.collect()
        # torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JindMulti')
    parser.add_argument('--list_target_batches', nargs='+', help='List of target batches')
    args = parser.parse_args()
    main(args)


#nohup python /home/jsanchoz/data/josebas/JIND_Iterative/JIND-continual_integration/jind_multi/main/Main_scamara.py --list_target_batches 'Bai_CR_basal' 'Bai_CR_CD19_cocult' 'Bai_CR_MESO_cocult' 'Bai_HD_CD19_cocult' 'Bai_HD_MESO_cocult' 'Bai_HD_basal' 'Bai_NR_basal' 'Bai_NR_CD19_cocult' 'Bai_NR_MESO_cocult' 'Bor_D1_28Z_CD19_Stim' 'Bor_D1_28Z_No_Stim' 'Bor_D1_BBZ_CD19_Stim' 'Bor_D1_BBZ_No_Stim' 'Bor_D1_Z_CD19_Stim' 'Bor_D1_Z_No_Stim' 'Bor_D2_28Z_CD19_Stim' 'Bor_D2_28Z_No_Stim' 'Bor_D2_BBZ_CD19_Stim' 'Bor_D2_BBZ_No_Stim' 'Bor_D2_Z_CD19_Stim' 'Bor_D2_Z_No_Stim' 'Den_Pt_14' 'Den_Pt_15' 'Den_Pt_16' 'Den_Pt_18' 'Den_Pt_20' 'Den_Pt_21' 'Den_Pt_26' 'Den_Pt_27' 'Den_Pt_28' 'Den_Pt_33' 'Den_Pt_34' 'Den_Pt_37' 'Den_Pt_38' 'Den_Pt_40' 'Den_Pt_41' 'Den_Pt_42' 'Den_Pt_43' 'Den_Pt_49' 'Den_Pt_50' 'Den_Pt_54' 'Den_Pt_55' 'Den_Pt_56' 'Den_Pt_59' 'Den_Pt_64' 'Goo_Pt_116' 'Goo_Pt_125' 'Goo_Pt_129' 'Goo_Pt_245' 'Goo_Pt_253' 'Goo_Pt_263' 'Goo_Pt_276' 'Goo_Pt_282' 'Lyn_Exp1_CD19' 'Lyn_Exp1_GD2' 'Lyn_Exp2_Cont' 'Lyn_Exp2_JUN' 'Mel_PT1_M12' 'Mel_PT1_M15' 'Mel_PT1_Y9' 'Mel_PT2_M3' 'Mel_PT2_Y3' 'Mel_PT2_Y6_5' 'She_CLL_1_d21' 'She_CLL_1_d38' 'She_CLL_1_d112' 'She_CLL_1_IP' 'She_CLL_2_d12' 'She_CLL_2_d29' 'She_CLL_2_d83' 'She_CLL_2_IP' 'She_NHL_6_d12' 'She_NHL_6_d29' 'She_NHL_6_d102' 'She_NHL_7_d12' 'She_NHL_7_d28' 'She_NHL_7_d89' 'She_NHL_7_IP' 'Wan_PD1' 'Wan_PD2' 'Wan_PD3' 'Wan_SPD1' 'Wan_SPD2' 'Wan_SPD3' 'Xha_Control' 'Xha_Raji_stim_1' 'Xha_Raji_stim_2' 'LiX_IP' 'LiX_PP' 'LiX_RP' 'Rod_D10' 'Rod_D10_d7' 'Rod_D14' 'Rod_D14_d7' 'Rod_D18' 'Rod_D18_d7' 'Har_Pat1_IP' 'Har_Pat2_D7' 'Har_Pat2_IP' 'Har_Pat3_IP' 'Har_Pat4_IP' 'Har_Pat5_IP' 'Har_Pat6_D7' 'Har_Pat7_D7' 'Har_Pat7_IP' 'Har_Pat8_IP' 'Har_Pat8_D7' 'Har_Pat9_IP' 'Har_Pat10_IP' 'Har_Pat10_D7' 'Har_Pat11_D7' 'Har_Pat12_D7' 'Har_Pat12_IP' 'Har_Pat12_D14' 'Har_Pat13_IP' 'Har_Pat14_D14' 'Har_Pat14_IP' 'Har_Pat15_D7' 'Har_Pat15_IP' 'Har_Pat16_D7' 'Har_Pat16_IP' 'Har_Pat18_IP' 'Har_Pat19_IP' 'Har_Pat20_IP' 'Har_Pat20_D14' 'Har_Pat21_D7' 'Har_Pat21_D14' 'Har_Pat21_IP' 'Har_Pat22_D7' 'Har_Pat22_IP' 'Har_Pat23_D7' 'Har_Pat23_IP' 'Har_Pat24_IP' 'Har_Pat24_D7' 'Har_Pat25_IP' 'Har_Pat25_D7' 'Har_Pat26_IP' 'Har_Pat26_D7' 'Har_Pat27_D7' 'Har_Pat27_IP' 'Har_Pat28_IP' 'Har_Pat28_D7' 'Har_Pat29_IP' 'Har_Pat29_IP_retreat' 'Har_Pat29_D7' 'Har_Pat29_D7_retreat' 'Har_Pat30_D7' 'Har_Pat30_IP' 'Har_Pat31_D7' 'Har_Pat31_IP' 'Har_Pat32_IP' 'Har_Pat32_D7' 'LiX_ac25' 'LiX_ac26' 'LiX_ac27' 'LiX_ac28' 'LiX_ac29' 'LiX_ac30' 'LiX_ac31' 'LiX_ac32' 'LiX_ac33' 'LiX_ac34' 'LiX_ac36' 'LiX_ac37' 'LiX_ac38' 'LiX_ac39' 'LiX_ac40' 'LiX_ac42' 'LiX_ac44' 'LiX_ac45' 'LiX_ac47' 'LiX_ac49' 'LiX_ac50' 'LiX_ac51' 'LiX_ac52' 'LiX_ac53' 'LiX_ac54' 'LiX_ac55' 'LiX_ac57' 'LiX_ac58' 'LiX_ac59' > out_scamara_restofthem.out &

