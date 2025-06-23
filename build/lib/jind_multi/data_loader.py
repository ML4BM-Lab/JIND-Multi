import scanpy as sc
import pandas as pd
import numpy as np
import warnings
import ast
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from .utils import dimension_reduction, preprocess, filter_cells
from .config_loader import get_config

def load_and_process_data(args, config={}):
    # Read and process the data
    config = get_config(config)['data']
    try:
        adata = sc.read(args.PATH)
        adata.var_names_make_unique()
    except OSError as e:
        raise RuntimeError(f"Failed to read the H5AD file at {args.PATH}. The file may be corrupted or incomplete.")
    # Filter and verify the data
    data = filter_and_verify_data(adata, args)
    # Process the data
    data = preprocess_data(data, config)
    return data

def filter_and_verify_data(adata, args):
    exclude_datasets_names = getattr(args, 'EXCLUDE_DATASETS_NAMES')
    if exclude_datasets_names:
        print(f'Excluding batches {exclude_datasets_names} from your data')
        exclude_list = ast.literal_eval(exclude_datasets_names)
        print(f'🔍 exclude_list evaluado: {exclude_list}')
        for i, val in enumerate(exclude_list):
            print(f"🔹 Elemento {i}: {val} (type: {type(val)})")
        selected_batches = list(dict.fromkeys(exclude_list))
        adata = adata[~adata.obs[args.BATCH_COL].isin(selected_batches)]
    
    # Convert to DataFrame and add columns
    data = adata.to_df()
    data['batch'] = adata.obs[args.BATCH_COL]
    data['labels'] = adata.obs[args.LABELS_COL]
    # Verify datasets contain sufficient entries
    check_dataset_entries(data, args)
    # Select common genes and labels
    data = select_common_genes_and_labels(data)
    return data

def check_dataset_entries(data, args):
    # Check if the source dataset has at least 2 entries if user set a source dataset
    if getattr(args, 'SOURCE_DATASET_NAME', None):
        source_entries = data[data['batch'] == args.SOURCE_DATASET_NAME]
        if source_entries.shape[0] <= 1:
            raise ValueError(f"Error: The source dataset {args.SOURCE_DATASET_NAME} has less than 2 entries. Please ensure this batch name is valid and the dataset contains enough samples.")
    
    # Check if the target dataset has at least 2 entries
    target_entries = data[data['batch'] == args.TARGET_DATASET_NAME]
    if target_entries.shape[0] <= 1:
        raise ValueError(f"Error: The target dataset {args.TARGET_DATASET_NAME} has less than 2 entries. Please ensure this batch name is valid and the dataset contains enough samples.")
    
    # Check if train datasets have at least 2 entries
    if getattr(args, 'INTER_DATASETS_NAMES', None):
        train_batches = ast.literal_eval(args.INTER_DATASETS_NAMES)
        for batch_name in train_batches:
            batch_entries = data[data['batch'] == batch_name]
            if batch_entries.shape[0] <= 1:
                raise ValueError(f"Error: The intermediate dataset {batch_name} has less than 2 entries. Please ensure this batch name is valid and the dataset contains enough samples.")
    
def select_common_genes_and_labels(data):
    batches = data['batch'].unique()
    common_genes = list(set.intersection(*[set(data[data['batch'] == batch].columns) for batch in batches]))
    common_genes.sort()
    data = data[list(common_genes)]
    common_labels = list(set.intersection(*[set(data[data['batch'] == batch]['labels']) for batch in batches]))
    common_labels.sort()
    data = data[data['labels'].isin(common_labels)]
    return data

def preprocess_data(data, config):
    data = preprocess(data, count_normalize=config['count_normalize'], log_transformation=config['log_transformation'])
    data = dimension_reduction(data, num_features=config['num_features'])
    data = filter_cells(data, min_cell_type_population=config['min_cell_type_population'], max_cells_for_dataset=config['max_cells_for_dataset'])
    data = data.reindex(sorted(data.columns), axis=1)  # Reorder columns
    return data

# 4) Plot umap before batch effect reduction
#sc.pl.scatter(adata, basis='umap', color=[labels_col, batch_col], frameon=False, show=False)
#plt.savefig("/".join(path.split('/')[0:4])+'/umap_batch_labels.png', bbox_inches="tight")

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

