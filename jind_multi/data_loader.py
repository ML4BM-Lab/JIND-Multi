import scanpy as sc
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
#from torch.utils.data import Dataset
from .utils import dimension_reduction, preprocess, filter_cells, create_scanpy_embeddings, create_scanpy_umap, create_umap_from_dataframe
from .config_loader import get_config

def load_and_process_data(args, config={}):
    # 1) Read ann object and add batch and labels columns
    config = get_config(config)['data']
    adata = sc.read(args.PATH)
    if args.TRAIN_DATASETS_NAMES:
        selected_batches = list(dict.fromkeys([args.SOURCE_DATASET_NAME] + ast.literal_eval(args.TRAIN_DATASETS_NAMES) + [args.TARGET_DATASET_NAME]))
        adata = adata[adata.obs[args.BATCH_COL].isin(selected_batches)]
    data = adata.to_df()
    data['batch'] = adata.obs[args.BATCH_COL]
    data['labels'] = adata.obs[args.LABELS_COL]
    # 2) Select commun genes and labels between batches
    batches = data['batch'].unique()
    common_genes = list(set.intersection(*[set(data[data['batch'] == batch].columns) for batch in batches]))
    common_genes.sort()
    data = data[list(common_genes)]
    common_labels = list(set.intersection(*[set(data[data['batch'] == batch]['labels']) for batch in batches]))
    common_labels.sort()
    data = data[data['labels'].isin(common_labels)]
    # 3) Processing
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

