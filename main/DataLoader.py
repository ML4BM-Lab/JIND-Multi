import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from Utils import dimension_reduction, preprocess, filter_cells, create_scanpy_embeddings, create_scanpy_umap, create_umap_from_dataframe
from ConfigLoader import get_config


# def generate_datasets_from_atac2():
#     path = get_config()['data']['atac2_data_path']
#     # Select tissues batch and labels for the experiment:
#     adata = sc.read_h5ad(path)
#     selected_tissues = ['heart_sample', 'liver_sample', 'kidney_sample', 'lung_sample', 'cerebrum_sample']
#     filtered_adata = adata[adata.obs['cell_id'].str.startswith(tuple(selected_tissues))]
    
#     for tissue in selected_tissues:
#         # Create a adata object for each of the tissues
#         tissue_adata = filtered_adata[filtered_adata.obs['cell_id'].str.startswith(tissue)]
#         # Write each tissue's adata to a separate H5AD file
#         path_save = '/home/jsanchoz/data/josebas/JIND_Iterative/JIND-continual_integration/jind_multi/resources/data/atac_2'
#         tissue_adata.write(f'{path[0:24]}/{tissue}_norm_scaled_data_annotated.h5ad')
    

def load_tissue_atac2_data_path(path):
    adata = sc.read_h5ad(path)
    data = adata.to_df()
    data["labels"] = adata.obs.celltype.values.tolist()
    data["batch"] = data.index.str.split('_').str[:3].str.join('_')
    batches = data['batch'].unique()
    common_genes = list(set.intersection(*[set(data[data['batch'] == batch].columns) for batch in batches]))
    common_genes.sort()
    data = data[list(common_genes)]
    common_labels = list(set.intersection(*[set(data[data['batch'] == batch]['labels']) for batch in batches]))
    common_labels.sort()
    data = data[data['labels'].isin(common_labels)]
    return data

    # heart: source: heart_sample_39, target: heart_sample_14; before min cell filtering: 35194 samples, 12436 genes and 20 labels
    # cerebrum: source: cerebrum_sample_6, target: cerebrum_sample_66, before min cell filtering: 69083 samples, 22 labels
    # kidney: source: kidney_sample_3, target: kidney_sample_67, before min cell filtering: 20853 samples, 26 labels
    # liver: source: liver_sample_35, target: liver_sample_9, before min cell filtering: 162168 samples, 12436 genes, 34 labels
    # lung: source: lung_sample_47, target: lung_sample_70, before min cell filtering: 54246 samples, 12436 genes, 20 labels

# for batch in batches:
#     print(batch, data[data.batch == batch].shape, len(data[data.batch == batch]['labels'].unique()))


def load_nsclc_lung_data_path(path={}):
    path = path if path is not None else get_config()['data']['nsclc_lung_data_path']
    adata = sc.read_h5ad(path)

    # Select batch and labels for the experiment:
    adata = adata[adata.obs['predicted_labels_majority'].isin(['B cells', 'CD4 T cells', 'CD8 T cells', 'Monocyte-derived Mph'])]
    data = adata.to_df()
    data['batch'] = adata.obs['Donor']
    data['labels'] = adata.obs['predicted_labels_majority']
    batches = data['batch'].unique()
    common_genes = list(set.intersection(*[set(data[data['batch'] == batch].columns) for batch in batches]))
    common_genes.sort()
    data = data[list(common_genes)]
    common_labels = list(set.intersection(*[set(data[data['batch'] == batch]['labels']) for batch in batches]))
    common_labels.sort()
    data = data[data['labels'].isin(common_labels)]

    sc.pl.scatter(adata, basis='umap', color=['predicted_labels_majority', 'batch'], frameon=False, show=False)
    plt.savefig("/".join(path.split('/')[0:4])+'/umap_batch_labels.png', bbox_inches="tight")
    return data

def load_human_brain_neurips_data(path={}):
    path = path if path is not None else get_config()['data']['human_brain_neurips_data_path']
    adata = sc.read(path)

    # Select batch and labels for the experiment:
    adata = adata[adata.obs['batch'].isin(['ADx1', 'ADx2', 'ADx4', 'AD2', 'C4', 'C7'])]
    adata = adata[adata.obs['label'].isin(['Astrocyte', 'BEC Arterial', 'BEC Capillary', 'BEC Venous', 'Oligodendrocyte', 'Pericyte', 'SMC'])]
    data = adata.to_df()
    data['batch'] = adata.obs['batch']
    data['labels'] = adata.obs['label']
    batches = data['batch'].unique()
    common_genes = list(set.intersection(*[set(data[data['batch'] == batch].columns) for batch in batches]))
    common_genes.sort()
    data = data[list(common_genes)]
    common_labels = list(set.intersection(*[set(data[data['batch'] == batch]['labels']) for batch in batches]))
    common_labels.sort()
    data = data[data['labels'].isin(common_labels)]

    sc.pl.scatter(adata, basis='umap', color=['label', 'batch'], frameon=False, show=False)
    plt.savefig("/".join(path.split('/')[0:4])+'/umap_batch_labels.png', bbox_inches="tight")
    return data

def load_test_data(path={}):
    path = path if path is not None else get_config()['data']['test_data_path']
    train_path = path + "train.pkl"
    test_path = path + "test.pkl"

    train_batch = pd.read_pickle(train_path)
    test_batch = pd.read_pickle(test_path)

    common_genes = list(set(train_batch.columns).intersection(set(test_batch.columns)))
    common_genes.sort()
    train_batch = train_batch[list(common_genes)]
    test_batch = test_batch[list(common_genes)]

    train_batch['batch'] = ["Source"] * train_batch.shape[0]
    test_batch['batch'] = ["Target"] * test_batch.shape[0]
    return train_batch.append(test_batch)

def load_pancreas_data(path={}): 
    path = path if path is not None else get_config()['data']['pancreas_data_path']
    adata = sc.read(path)
    # Select batch and labels for the experiment:
    adata = adata[adata.obs['celltype'].isin(['acinar', 'alpha', 'beta', 'delta', 'ductal', 'gamma'])]
    data = adata.to_df()
    data['batch'] = adata.obs['batch']
    data['labels'] = adata.obs['celltype']
    
    batches = data['batch'].unique()
    common_genes = list(set.intersection(*[set(data[data['batch'] == batch].columns) for batch in batches]))
    common_genes.sort()
    data = data[list(common_genes)]
    common_labels = list(set.intersection(*[set(data[data['batch'] == batch]['labels']) for batch in batches]))
    common_labels.sort()
    data = data[data['labels'].isin(common_labels)]

    sc.pl.scatter(adata, basis='umap', color=['celltype', 'batch'], frameon=False, show=False)
    plt.savefig("/".join(path.split('/')[0:4])+'/umap_batch_labels.png', bbox_inches="tight")
    return data


def load_bmmc_gex_from_atac_data(path={}):
    path = path if path is not None else get_config()['data']['bmmc_atac_path']
    adata = sc.read_h5ad(path)
    # Select batch and labels for the experiment:
    adata = adata[~adata.obs['batch'].isin(['s3d6', 's3d7', 's4d9'])]
    adata = adata[adata.obs['cell_type'].isin(['Erythroblast', 'Proerythroblast', 'CD14+ Mono', 'Naive CD20+ B', 'NK', 'CD8+ T', 'CD4+ T activated'])]
    data = adata.to_df()
    data['labels'] = adata.obs['cell_type']
    data['batch'] = adata.obs['batch']
    batches = data['batch'].unique()
    # Remove repeated genes
    data = data.loc[:, ~data.columns.duplicated()]
    common_genes = list(set.intersection(*[set(data[data['batch'] == batch].columns) for batch in batches]))
    common_genes.sort()
    data = data[list(common_genes)]
    common_labels = list(set.intersection(*[set(data[data['batch'] == batch]['labels']) for batch in batches]))
    common_labels.sort()
    data = data[data['labels'].isin(common_labels)]
    sc.pl.scatter(adata, basis='umap', color=['cell_type', 'batch'], frameon=False, show=False)
    plt.savefig("/".join(path.split('/')[0:4])+'/ATAC_plots/umap_batch_labels.png', bbox_inches="tight")
    return data

def load_bmmc_gex_data(path={}): 
    path = path if path is not None else get_config()['data']['bmmc_gex_path']
    adata = sc.read_h5ad(path)
    # Select batch and labels for the experiment:
    adata = adata[~adata.obs['batch'].isin(['s3d6', 's3d7', 's4d9'])]
    adata = adata[adata.obs['cell_type'].isin(['Erythroblast', 'Proerythroblast', 'CD14+ Mono', 'Naive CD20+ B', 'NK', 'CD8+ T', 'CD4+ T activated'])]
    data = adata.to_df()
    data['labels'] = adata.obs['cell_type']
    data['batch'] = adata.obs['batch']
    batches = data['batch'].unique()
    # Remove repeated genes
    data = data.loc[:, ~data.columns.duplicated()]
    common_genes = list(set.intersection(*[set(data[data['batch'] == batch].columns) for batch in batches]))
    common_genes.sort()
    data = data[list(common_genes)]
    common_labels = list(set.intersection(*[set(data[data['batch'] == batch]['labels']) for batch in batches]))
    common_labels.sort()
    data = data[data['labels'].isin(common_labels)]
    sc.pl.scatter(adata, basis='umap', color=['cell_type', 'batch'], frameon=False, show=False)
    plt.savefig("/".join(path.split('/')[0:4])+'/GEX_plots/umap_batch_labels.png', bbox_inches="tight")
    return data

def load_bmmc_omics_data(config):
    data_atac = load_bmmc_gex_from_atac_data(config['bmmc_atac_path'])
    data_gex =  load_bmmc_gex_data(config['bmmc_gex_path'])
    # add atac and gex info to batch column and index
    data_atac['batch'] = data_atac.batch.astype(str)+'_atac'
    data_atac.index = data_atac.index+'_atac'
    data_gex['batch'] = data_gex.batch.astype(str)+'_gex'
    data_gex.index = data_gex.index+'_gex'
    # common genes
    common_genes = list(data_atac.columns.intersection(data_gex.columns))
    common_genes.sort()
    data_atac = data_atac.loc[:,common_genes]
    data_gex = data_gex.loc[:,common_genes]
    data = pd.concat([data_atac, data_gex])
    return data



def load_data(data_type, config={}, source_dataset_name=None, preserve_target_labels_dataset_name=None):
    # Read Data
    config = get_config(config)['data']
    if data_type == "pancreas":
        data = load_pancreas_data(config['pancreas_data_path'])
    elif data_type == "breast":
        data = load_brca_data(config['brca_data_path'])
    elif data_type == "brain":
        data = load_brain_data(config['brain_data_source'], config['brain_data_source_gene_nums'])
    elif data_type == "brain_neurips":
        data = load_human_brain_neurips_data(config['human_brain_neurips_data_path'])
    elif data_type == "nsclc_lung":
        data = load_nsclc_lung_data_path(config['nsclc_lung_data_path'])
    elif data_type == "bmmc_atac":
        data = load_bmmc_gex_from_atac_data(config['bmmc_atac_path'])
    elif data_type == "bmmc_omics":
        data = load_bmmc_omics_data(config)
    elif data_type in ["heart_atac2", "cerebrum_atac2", "kidney_atac2", "liver_atac2", "lung_atac2"]:
        tissue = data_type.split('_')[0] 
        config2 = config['tissue_data_path'].replace('tissue', tissue)
        data = load_tissue_atac2_data_path(config2)
    else:
        data = load_test_data(config['test_data_path'])
   
    # Processing
    data = preprocess(data, count_normalize=config['count_normalize'], log_transformation=config['log_transformation'])
    data = dimension_reduction(data, num_features=config['num_features'])
    data = filter_cells(data, min_cell_type_population=config['min_cell_type_population'], max_cells_for_dataset=config['max_cells_for_dataset'],
                        source_dataset_name=source_dataset_name, preserve_target_labels_dataset_name=preserve_target_labels_dataset_name)

    data = data.reindex(sorted(data.columns), axis=1)  # Reorder columns
    return data

class DataLoaderCustom(Dataset):
    def __init__(self, features, labels=None, weights=None, transform=None):
        """
            Args:
                features (string): np array of features.
                transform (callable, optional): Optional transform to be applied
                    on a sample.
        """
        self.features = features
        self.labels = labels
        self.weights = weights
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = {}
        sample['x'] = self.features[idx].astype('float32')
        if self.labels is not None:
            sample['y'] = self.labels[idx]
            if self.weights is not None:
                sample['w'] = self.weights[self.labels[idx]].astype('float32')
        return sample


# def load_brca_data(path={}): 
#     path = path if path is not None else get_config()['data']['brca_data_path']
#     adata = sc.read(path)
#     data = adata.to_df()
#     data['batch'] = adata.obs['orig.ident']
#     data['labels'] = adata.obs['celltype_minor'] # options: celltype_subset, celltype_minor, celltype_major

#     batches = data['batch'].unique()
#     common_genes = list(set.intersection(*[set(data[data['batch'] == batch].columns) for batch in batches]))
#     common_genes.sort()
#     data = data[list(common_genes)]
#     return data    

# def load_brain_data():
#     data = pd.read_csv("drive/MyDrive/Colab Notebooks/JIND/data/Brain/MouseV1_MouseALM_HumanMTG.csv", index_col='Unnamed: 0')
#     data['labels'] = list(pd.read_csv("drive/MyDrive/Colab Notebooks/JIND/data/Brain/MouseV1_MouseALM_HumanMTG_Labels34.csv")['x'])
#     data['batch'] = ['MouseV1'] * 12552 + ['MouseALM'] * 8128 + ['HumanMTG'] * 14055

#     # Cell types which have more than 200 cells in each dataset
#     data = data[data['labels'].isin( ['Exc L2/3 IT', 'Exc L3/5 IT', 'Exc L4/5 IT', 'Exc L5/6 NP', 'Exc L6 CT', 'Lamp5 Rosehip', 'Pvalb 2', 'Sst 1', 'Sst 5', 'Vip 4'])]

#     batches = data['batch']
#     labels = data['labels']
#     data = data.drop(['batch', 'labels'], axis=1)
#     features = data.columns[np.argsort(-np.var(data.values, axis=0))[:5000]]
#     data = data[features]
#     data['batch'] = batches
#     data['labels'] = labels
#     return data