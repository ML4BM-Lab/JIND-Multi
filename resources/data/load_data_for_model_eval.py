import numpy as np
from jind_multi.resources.data import load_PBMC_data
from jind_multi.resources.data import load_PBMCprotocol_data
from jind_multi.main.Utils import dimension_reduction, preprocess, filter_cells

def load_pbmc_batch1_batch2_data():
    path = 'drive/MyDrive/Jind/jind_multi/resources/data'
    PBMC_batch1_data = load_PBMC_data.load_PBMC_batch1_data(path)
    PBMC_batch1_data = load_PBMC_data.curate_PBMC_demulx_celltypes(PBMC_batch1_data, 0)
    batch1_data = PBMC_batch1_data.to_df()
    batch1_data['labels'] = PBMC_batch1_data.obs['cell.type']
    batch1_data['batch'] = ['Batch1'] * len(batch1_data)
    # batch1_data['batch'] = [str(x) for x in PBMC_batch1_data.obs['ind']]
    PBMC_batch1_data = None

    PBMC_batch2_data = load_PBMC_data.load_PBMC_batch2_data(path, 'control')
    PBMC_batch2_data = load_PBMC_data.curate_PBMC_demulx_celltypes(PBMC_batch2_data, 0)
    batch2_data_cont = PBMC_batch2_data.to_df()
    batch2_data_cont['labels'] = PBMC_batch2_data.obs['cell.type']
    # batch2_data_cont['batch'] = [str(x)+'_c' for x in PBMC_batch2_data.obs['ind']]
    # batch2_data_cont = batch2_data_cont[batch2_data_cont['batch']!='1039_c']
    # batch2_data_cont = batch2_data_cont[batch2_data_cont['batch']!='107_c']
    batch2_data_cont['batch'] = ['Batch2_control'] * len(batch2_data_cont)
    PBMC_batch2_data = None

    PBMC_batch2_data = load_PBMC_data.load_PBMC_batch2_data(path, 'stimulated')
    PBMC_batch2_data = load_PBMC_data.curate_PBMC_demulx_celltypes(PBMC_batch2_data, 0)
    batch2_data_stim = PBMC_batch2_data.to_df()
    batch2_data_stim['labels'] = PBMC_batch2_data.obs['cell.type']
    # batch2_data_stim['batch'] = [str(x)+'_s' for x in PBMC_batch2_data.obs['ind']]
    # batch2_data_stim = batch2_data_stim[batch2_data_stim['batch']!='1039_s']
    # batch2_data_stim = batch2_data_stim[batch2_data_stim['batch']!='107_s']
    batch2_data_stim['batch'] = ['Batch2_stim'] * len(batch2_data_stim)
    PBMC_batch2_data = None

    batch2_data = batch2_data_stim.append(batch2_data_cont)
    batch2_data_stim = None
    batch2_data_cont = None

    batch2_data = batch2_data.loc[:, (batch2_data != 0).any(axis=0)]
    batch1_data = batch1_data.loc[:, (batch1_data != 0).any(axis=0)]
    common_genes = list(set(batch1_data.columns).intersection(set(batch2_data.columns)))
    common_genes.sort()
    batch1_data = batch1_data[list(common_genes)]
    batch2_data = batch2_data[list(common_genes)]

    data = batch1_data.append(batch2_data)
    batch1_data = None
    batch2_data = None

    batch = data['batch']
    labels = data['labels']
    features = data.columns[np.argsort(-np.var(data.drop(['batch', 'labels'], axis=1).values, axis=0))[:5000]]
    data = data[features]
    data['batch'] = batch
    data['labels'] = labels

    data = filter_cells(data, min_cell_type_population=30)
    return data

def load_pbmc_protocol_data():
    path = 'drive/MyDrive/Jind/jind_multi/resources/data'
    adata = load_PBMCprotocol_data.load_PBMC_protocols_data(path, curate=True)

    protocol_data = adata.to_df()
    protocol_data['labels'] = adata.obs['cell.type']
    protocol_data['labels'] = [w.replace('CD16+ monocyte', 'Monocytes') for w in protocol_data['labels']]
    protocol_data['labels'] = [w.replace('Plasmacytoid dendritic cell', 'Dendritic cell') for w in
                               protocol_data['labels']]
    protocol_data = protocol_data[protocol_data['labels'] != 'Unassigned']
    protocol_data['batch'] = adata.obs['Method']
    protocol_data['batch'] = [w.replace('CEL-Seq2', 'CEL_Smart') for w in protocol_data['batch']]
    protocol_data['batch'] = [w.replace('Smart-seq2', 'CEL_Smart') for w in protocol_data['batch']]
    protocol_data = protocol_data.loc[:, (protocol_data != 0).any(axis=0)]

    data = protocol_data
    batch = data['batch']
    labels = data['labels']
    features = data.columns[np.argsort(-np.var(data.drop(['batch', 'labels'], axis=1).values, axis=0))[:5000]]
    data = data[features]
    data['batch'] = batch
    data['labels'] = labels

    data = filter_cells(data, min_cell_type_population=30)
    return data