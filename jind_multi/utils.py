import scanpy as sc
import numpy as np
from itertools import product
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import math
import re
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from .models import Classifier, ClassifierBig
import json
import torch
from .config_loader import get_config

def find_saved_models(path_saved_models, train_data):
    # List to store the paths of saved model files
    file_paths = []
    train_batches = set(train_data.batch.unique())
    # Iterate over directories within path_saved_models
    for root, dirs, files in os.walk(path_saved_models):
        # Iterate over files in the current directory
        for file in files:
            # Check if the file starts with any of the names in the start_names set
            if any(file.startswith(name) for name in train_batches) and file.endswith('.pt'):
                file_paths.append(os.path.join(root, file))
    return file_paths

def load_trained_models(file_paths, train_data, source_dataset_name, config={}):
    use_cuda = get_config(config)['train_classifier']['cuda']
    device = torch.device("cuda" if use_cuda else "cpu")
    model = {}
    print("\n[load_trained_models] Already trained models found in file_paths.")
    # Load the Source Classifier Model
    print("[load_trained_models] Load the Source Classifier Model")
    print(f"[load_trained_models] Trained Classifier Dataset Name: {source_dataset_name}")
    model_path = next((path for path in file_paths if source_dataset_name in path), None)
    base_model = Classifier(
            train_data.drop(['batch', 'labels'], axis=1).shape[1], 256, 1500, 
            len(list(set(train_data[train_data.batch == source_dataset_name]['labels'])))).to(device)
    base_model.load_state_dict(torch.load(model_path))
    # Add loaded model to saved model dictionary
    print("[load_trained_models] Add loaded model to saved model dictionary")
    model[source_dataset_name] = base_model
    # Load the rest of intermediate models
    print("[load_trained_models] Load the rest of intermediate models")
    train_batches = set(train_data.batch.unique())
    for dataset in [d for d in train_batches if d!=source_dataset_name]:
        print(f"[load_trained_models] Trained Intermediate model name: {dataset}")
        model_path = next((path for path in file_paths if dataset in path), None)
        model_copy = ClassifierBig(
                    base_model, 
                    train_data[train_data['batch'] == dataset].shape[1]-2, 
                    256, 512).to(device)
        model_copy.load_state_dict(torch.load(model_path))
        # Add loaded model to saved model dictionary
        print("[load_trained_models] Add loaded model to saved model dictionary")
        model[dataset] = model_copy
    return model

def load_val_stats(start_dir, target_file):
    for dirpath, dirnames, filenames in os.walk(start_dir):
        if target_file in filenames:
            file_path = os.path.join(dirpath, target_file)
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    # Convert lists to NumPy arrays
                    data['pred'] = np.array(data['pred'])
                    data['true'] = np.array(data['true'])
                    return data
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON file: {e}")
                    return None
    return None

def remove_pth_files(directory):
    if os.path.exists(directory):
        # Iterate over all files in the directory
        for filename in os.listdir(directory):
            # Check if the file has the .pth extension
            if filename.endswith('.pth'):
                file_path = os.path.join(directory, filename)
                try:
                    # Remove the file
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
    else:
        print(f"The directory {directory} does not exist.")

def create_scanpy_embeddings(adata, basis, batch, labels, path):
    with plt.rc_context({"figure.figsize": (7, 6), "figure.dpi": (300), "font.size": 4}): #"font.size": 4
        fig1 = sc.pl.embedding(adata, basis=basis, color = labels, legend_loc='on data', legend_fontsize = 'medium', return_fig = True)
        fig2 = sc.pl.embedding(adata, basis=basis, color = batch, legend_fontsize = 'small', return_fig = True)
        for ax in fig1.axes:
            ax.xaxis.label.set_fontsize(10)
            ax.yaxis.label.set_fontsize(10)
        for ax in fig2.axes:
            ax.xaxis.label.set_fontsize(10)
            ax.yaxis.label.set_fontsize(10)
        plt.figure(fig1.number) 
        plt.title('U-MAP of the gene expression of the batches by label', fontsize=12) 
        plt.figure(fig2.number)
        plt.title('U-MAP of the gene expression of the batches', fontsize=12) 
        fig1.savefig(path+f'/{basis}_label_initial.png')
        fig2.savefig(path+f'/{basis}_batch_initial.png')
        plt.close()

def create_scanpy_umap(adata, batch, labels, path):
    with plt.rc_context({"figure.figsize": (7, 6), "figure.dpi": (300), "font.size": 5}): #"font.size": 4
        fig = sc.pl.umap(adata, color=[batch, labels], return_fig=True)
        for ax in fig.axes:
            ax.xaxis.label.set_fontsize(10)
            ax.yaxis.label.set_fontsize(10)
        fig.savefig(path+'/umap_initial.png')
        plt.close()

def create_umap_from_dataframe(dataframe, batch_col, labels_col, path):
    df = dataframe.copy()
    batch_labels = df[[batch_col, labels_col]]
    df.drop([batch_col, labels_col], axis=1, inplace=True)
    pca = PCA(n_components=50)
    reduced_feats = pca.fit_transform(df)
    emb = TSNE(n_components=2, verbose=1, n_jobs=-1, perplexity=50, random_state=43).fit_transform(reduced_feats)
    df2 = pd.DataFrame({'tSNE_x': emb[:, 0], 'tSNE_y': emb[:, 1], 'Labels': batch_labels[labels_col], 'Batch': batch_labels[batch_col]})
    plot_and_save_tsne(df2, path, plot_name='tsne_common_labels_initial_experiment_samples')


def preprocess(data, count_normalize=True, log_transformation=True, target_sum=1e4):
    batches = data['batch']
    labels = data['labels']
    raw_features = data.drop(['batch', 'labels'], axis=1).values

    error = np.mean(np.abs(raw_features - np.rint(raw_features)))
    if error != 0:
        return data
    if count_normalize:
        print('[Utils] Normalizing counts ...')
        raw_features = raw_features / (np.sum(raw_features, axis=1, keepdims=True) + 1e-5) * target_sum
    if log_transformation:
        print('[Utils] Applying log transformation ...')
        raw_features = np.log(1 + raw_features)

    # data = pd.DataFrame(raw_features, index=data.index, columns=list(set(data.columns) - {'batch', 'labels'}))
    data = pd.DataFrame(raw_features, index=data.index, columns=data.drop(['batch', 'labels'], axis=1).columns)

    data['batch'] = batches
    data['labels'] = labels
    return data


def dimension_reduction(data, num_features=5000):
    print('[Utils] Variance based dimension reduction ...')
    batches = data['batch']
    labels = data['labels']
    data = data.drop(['batch', 'labels'], axis=1)

    features = data.columns[np.argsort(-np.var(data.values, axis=0))[:num_features]]
    data = data[features]
    data['batch'] = batches
    data['labels'] = labels
    return data


def filter_cells(data, min_cell_type_population=100, max_cells_for_dataset=50000):
    # min_cell_type_population and max_cells_for_dataset filter (if one batch has one cell_type with a too low population these samples will be removed too from the other batches)
    batch_data = data[data['batch'] == data['batch'][0]]
    data_trimmed = batch_data[:max_cells_for_dataset]
    # keep only the first 50K cells for each batch
    for batch in [n for n in set(data['batch']) if n != data['batch'][0]]:
        batch_data = data[data['batch'] == batch]
        data_trimmed = data_trimmed.append(batch_data[:max_cells_for_dataset])

    data = data_trimmed
    cell_types_with_low_population = []
    for batch in set(data['batch']):
        cell_type_names, cell_type_counts = np.unique(data[data['batch'] == batch]['labels'], return_counts=True)
        for i in range(len(cell_type_names)):
            if cell_type_counts[i] < min_cell_type_population:
                cell_types_with_low_population = cell_types_with_low_population + [cell_type_names[i]]
                print("[Utils][filter_cells] Batch '{}': {} only has {} cells (min cell type population = {})".format(
                    batch, cell_type_names[i], cell_type_counts[i], min_cell_type_population))

    cell_types_to_retain = set(data['labels']) - set(cell_types_with_low_population)
    data = data[data['labels'].isin(cell_types_to_retain)]
    print("[Utils][filter_cells] Cell type population count in data: ", *np.unique(data['labels'], return_counts=True))
    for batch in set(data['batch']):
        print("[Utils] {}: ".format(batch), data[data['batch'] == batch].shape, *np.unique(data[data['batch'] == batch]['labels'], return_counts=True))
    return data

class ConfusionMatrixPlot:

    def __init__(self, confusion_matrix, display_true_labels, display_labels, title=''):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels.copy()
        self.displabelsx = display_labels.copy() # pred labels
        if "Novel" in display_labels:
            self.displabelsx.remove("Novel")
        self.displabelsy = display_true_labels.copy()
        # self.displabelsy.remove("Unassigned") 
        self.title = title

    def plot(self, include_values=True, cmap='viridis', xticks_rotation='vertical', values_format=None, ax=None, fontsize=10):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        self.text_ = None
        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)
            values_format = '.2g' if values_format is None else values_format
            # print text with appropriate color depending on background
            thresh = (cm.max() + cm.min()) / 2.0
            for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
                color = cmap_max if cm[i, j] < thresh else cmap_min
                self.text_[i, j] = ax.text(j, i, format(cm[i, j], values_format), ha="center", va="center", color=color, fontsize=fontsize)

        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]))
        ax.set_title("\n".join(self.title.split(".pdf")), fontsize=fontsize + 2)
        ax.set_xticklabels(self.displabelsx[:cm.shape[1]], fontsize=fontsize)
        ax.set_yticklabels(self.displabelsy[:cm.shape[0]], fontsize=fontsize)
        ax.set_ylabel(ylabel="True label", fontsize=fontsize + 2)
        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self


def plot_timeline(plot_embeddings, path, timeline_plot_name):
    plot_names = list(plot_embeddings.keys())
    plot_only_cmatrix = False if 'tsne' in plot_embeddings[plot_names[0]].keys() else True
    num_cols = 1 if plot_only_cmatrix else 3
    factor = max(1, len(plot_embeddings[plot_names[0]]['matrix'].confusion_matrix) // 10)

    fig, axes = plt.subplots(len(plot_names), num_cols, figsize=(10 * factor * num_cols, 8 * factor * len(plot_names)))
    fig.suptitle(timeline_plot_name, fontsize=20)

    i = -1
    for plot_name in plot_names:
        i = i + 1
        confusion_matrix = plot_embeddings[plot_name]['matrix']
        print("[JindLib]", plot_name, confusion_matrix.title)

        if plot_only_cmatrix:
            fig1 = plt.figure(figsize=(10 * factor, 8 * factor))
            confusion_matrix.plot(values_format='0.2f', ax=axes[i])
        else:
            df = plot_embeddings[plot_name]['tsne']
            batches = sorted(list(set(df['Batch'])))
            sns.scatterplot(x='tSNE_x', y='tSNE_y', hue='Batch', data=df, hue_order=batches, s=80, ax=axes[i, 0]).set_title(plot_name)
            sns.scatterplot(x='tSNE_x', y='tSNE_y', hue='Labels', data=df, hue_order=list(set(df['Labels'])), style='Batch', style_order=batches, s=80, ax=axes[i, 1]).set_title(plot_name)
            fig1 = plt.figure(figsize=(10 * factor, 8 * factor))
            confusion_matrix.plot(values_format='0.2f', ax=axes[i, 2])

    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.savefig("{}/{}.pdf".format(path, timeline_plot_name))
    plt.close(fig1)

def plot_cmat_timeline(conf_matrix_list, path, timeline_plot_name, num_datasets=1, cmat_print_counts=False):
    # conf_matrix_list is list of maps of confusion matrix. [('dataset1_initial',conf_mat1), ('dataset1_GAN',conf_mat2)]
    conf_matrices = [cmat for name, cmat in conf_matrix_list]
    plot_categories = [name.split('_').pop() for name, cmat in conf_matrix_list]
    
    num_plots = len(conf_matrix_list)
    num_cols = num_datasets
    num_rows = math.ceil(num_plots / num_cols)
    
    factor = max(1, len(conf_matrices[0].confusion_matrix) // 10)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10 * factor * num_cols, 8 * factor * num_rows))
    fig.suptitle(timeline_plot_name, fontsize=20)

    for i, (plot_name, confusion_matrix) in enumerate(conf_matrix_list):
        print("[Utils]", plot_name, confusion_matrix.title)
        row = i // num_cols
        col = i % num_cols
        confusion_matrix.plot(values_format='0.0f' if cmat_print_counts else '0.2f', ax=axes[row, col])
        target_title = confusion_matrix.title 
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.savefig("{}/{}.pdf".format(path, timeline_plot_name))
    plt.close(fig)
    raw_acc_per = re.search(r'raw (\d+\.\d+)%', target_title).group(1)
    eff_acc_per = re.search(r'eff (\d+\.\d+)%', target_title).group(1)
    mAP_per = re.search(r'mAP (\d+\.\d+)%', target_title).group(1)
    rejected_per = re.search(r'Rej \d+ \((\d+\.\d+)%\)', target_title).group(1)
    return raw_acc_per, eff_acc_per, mAP_per, rejected_per

def plot_and_save_tsne(dataframe, path, plot_name):
    df = dataframe
    
    fig_size = (14, 12)   
    marker_size = 100   
    legend_fontsize = 11 
    marker_scale = 1.5   

    # Plot genes of the batches in different colours
    if 'Labels' in df.columns:
        plt.figure(figsize=fig_size)
        order = sorted(list(set(df['Labels'])))
        print(f'Order:: {order}')
        batches = sorted(list(set(df['Batch'])))
        print(f'batches:: {batches}')
        g = sns.scatterplot(x='tSNE_x', y='tSNE_y', hue='Labels', data=df, hue_order=order, style='Batch', style_order=batches, s=marker_size)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legend_fontsize, markerscale=marker_scale)
        plt.title("{}_labels".format(plot_name), fontsize=legend_fontsize + 2)
        plt.tight_layout()
        plt.savefig("{}/{}_labels.pdf".format(path, plot_name))
        plt.close()

    # Plot batches in different colours
    if 'Batch' in df.columns:
        plt.figure(figsize=fig_size)
        hue_order = sorted(list(set(df['Batch'])), key=str.casefold)
        g = sns.scatterplot(x='tSNE_x', y='tSNE_y', hue='Batch', data=df, hue_order=hue_order, s=marker_size)
        plt.legend(fontsize=legend_fontsize, markerscale=marker_scale)
        plt.title("{}_batches".format(plot_name), fontsize=legend_fontsize + 2)
        plt.tight_layout()
        plt.savefig("{}/{}_batches.pdf".format(path, plot_name))
        plt.close()

    # Plot predictions in different colours
    if 'Raw_Predictions' in df.columns and 'Assignment' in df.columns and 'Evaluation' in df.columns:
        df['Evaluation'] = pd.Categorical(df['Evaluation'], categories=['Correct', 'Miss'], ordered=True) # NEW
        markers_dict = {'Correct': 'o', 'Miss': 'X'}
        # markers_list = ['o', 'X']  
        unique_assignments = df['Assignment'].unique()
        num_unique_assignments = len(unique_assignments)
        sizes_list = [marker_size, marker_size/3][:num_unique_assignments]

        plt.figure(figsize=fig_size)
        g = sns.scatterplot(
            x='tSNE_x', y='tSNE_y', 
            hue='Raw_Predictions', 
            style='Evaluation',   
            markers=markers_dict, #markers_list, NEW
            size='Assignment',  
            sizes=sizes_list, 
            data=df
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legend_fontsize, markerscale=marker_scale)
        plt.title("{}".format(plot_name), fontsize=legend_fontsize + 2)
        plt.tight_layout()
        plt.savefig("{}/{}.pdf".format(path, plot_name))
        plt.close()

def compute_ap(gts, preds):
    aps = []
    for i in range(preds.shape[1]):
        ap, prec, rec = calc_pr(gts == i, preds[:, i:i + 1])
        aps.append(ap)
    aps = np.array(aps)
    return np.nan_to_num(aps)


def calc_pr(gt, out, wt=None):
    gt = gt.astype(np.float64).reshape((-1, 1))
    out = out.astype(np.float64).reshape((-1, 1))

    tog = np.concatenate([gt, out], axis=1) * 1.
    ind = np.argsort(tog[:, 1], axis=0)[::-1]
    tog = tog[ind, :]
    cumsumsortgt = np.cumsum(tog[:, 0])
    cumsumsortwt = np.cumsum(tog[:, 0] - tog[:, 0] + 1)
    prec = cumsumsortgt / (cumsumsortwt + 1e-8)
    rec = cumsumsortgt / (np.sum(tog[:, 0]) + 1e-8)
    ap = voc_ap(rec, prec)
    return ap, rec, prec


def voc_ap(rec, prec):
    rec = rec.reshape((-1, 1))
    prec = prec.reshape((-1, 1))
    z = np.zeros((1, 1))
    o = np.ones((1, 1))
    mrec = np.vstack((z, rec, o))
    mpre = np.vstack((z, prec, z))

    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    I = np.where(mrec[1:] != mrec[0:-1])[0] + 1;
    ap = np.sum((mrec[I] - mrec[I - 1]) * mpre[I])
    return ap


def normalize(cm, normalize=None, epsilon=1e-8):
    with np.errstate(all='ignore'):
        if normalize == 'true': # Divide cada elemento de la matriz de confusión por la suma de las filas correspondientes. Esto normaliza las filas de la matriz, lo que significa que cada fila suma 1.
            cm = cm / (cm.sum(axis=1, keepdims=True) + epsilon)
        elif normalize == 'pred': #  Divide cada elemento de la matriz de confusión por la suma de las columnas correspondientes. Esto normaliza las columnas de la matriz, lo que significa que cada columna suma 1.
            cm = cm / (cm.sum(axis=0, keepdims=True) + epsilon)
        elif normalize == 'all':
            cm = cm / (cm.sum() + epsilon) # Divide cada elemento de la matriz de confusión por la suma total de todos los elementos de la matriz. Esto normaliza toda la matriz, lo que significa que la suma de todos los elementos de la matriz es 1.
        cm = np.nan_to_num(cm) # Esta línea reemplaza cualquier valor NaN en la matriz de confusión con ceros.
    return cm
