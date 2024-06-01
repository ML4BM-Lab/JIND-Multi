import numpy as np
import torch, sys, os, pdb, os.path
import json
import pandas as pd
from torch import optim
from torch.autograd import Variable
from Utils import ConfusionMatrixPlot, compute_ap, normalize, plot_and_save_tsne
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from Models import Classifier, Discriminator, ClassifierBig
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
from ConfigLoader import get_config
from DataLoader import DataLoaderCustom

class JindLib:
    global MODEL_WIDTH, LDIM, GLDIM, BATCH, LABELS
    BATCH = 'batch'
    LABELS = 'labels'
    MODEL_WIDTH = 1500
    LDIM = 256
    GLDIM = 512

    def __init__(self, source_labelled_data, output_path, config=get_config(), model=None, val_stats=None):
        gene_mat = source_labelled_data.drop([BATCH, LABELS], axis=1)
        cell_labels = source_labelled_data[LABELS]
        self.source_dataset_name = source_labelled_data[BATCH][0]
        self.class2num = None
        self.num2class = None
        self.reduced_features = None
        self.reduce_method = None

        if model is not None:
            if val_stats is None:
                raise ValueError("If you provide an already trained model, you must also provide val_stats for threshold calculation.")
            self.model = model
            self.val_stats = val_stats
        else:
            self.model = {} # Initial setup
            self.val_stats = None

        self.count_normalize = False
        self.logt = False
        self.path = output_path 
        os.system('mkdir -p {}'.format(self.path))
        os.makedirs(self.path, exist_ok=True)

        self.raw_features = gene_mat.values
        self.cell_ids = list(gene_mat.index)
        self.gene_names = list(gene_mat.columns)

        classes = list(set(cell_labels))
        classes.sort()
        self.classes = classes
        self.n_classes = len(classes)

        self.class2num = class2num = {c: i for (i, c) in enumerate(classes)}
        self.class2num['Unassigned'] = self.n_classes

        self.num2class = num2class = {i: c for (i, c) in enumerate(classes)}
        self.num2class[self.n_classes] = 'Unassigned'

        self.labels = np.array([class2num[i] for i in cell_labels])
        self.scaler = None
        self.config = get_config(user_conf=config)
        self.plot_embeddings = {}
        self.dataset_train_val = {}
        self.conf_matrix = []

    def preprocess(self, count_normalize=False, target_sum=1e4, logt=True):
        self.logt = logt
        self.count_normalize = count_normalize
        if count_normalize:
            print('Normalizing counts ...')
            self.raw_features = self.raw_features / (
                    np.sum(self.raw_features, axis=1, keepdims=True) + 1e-5) * target_sum
        if logt:
            print('Applying log transformation ...')
            self.raw_features = np.log(1 + self.raw_features)

    def dim_reduction(self, num_features=5000, method='var', save_as=None):
        mat = self.raw_features
        mat_round = np.rint(mat)
        error = np.mean(np.abs(mat - mat_round))
        if error == 0:
            self.preprocess(count_normalize=True, logt=True)

        dim_size = num_features
        self.reduce_method = method

        if method.lower() == 'pca':
            print('Performing PCA ...')
            self.pca = PCA(n_components=dim_size)
            self.reduced_features = self.pca.fit_transform(self.raw_features)
            if save_as is not None:
                np.save('{}_{}'.format(save_as, method), self.reduced_features)

        elif method.lower() == 'var':
            print('[JindLib] Performing Variance based reduction ...')
            self.variances = np.argsort(-np.var(self.raw_features, axis=0))[:dim_size]
            self.reduced_features = self.raw_features[:, self.variances]
            self.selected_genes = [self.gene_names[i] for i in self.variances]
            if save_as is not None:
                np.save('{}_{}'.format(save_as, method), self.reduced_features)

    def train_classifier(self, config=get_config()['train_classifier'], cmat=True):
        features = self.reduced_features if self.reduced_features is not None else self.raw_features
        labels = self.labels

        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        torch.backends.cudnn.deterministic = True

        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=config['val_frac'], stratify=labels, shuffle=True, random_state=config['seed'])
        train_dataset = DataLoaderCustom(X_train, y_train)
        val_dataset = DataLoaderCustom(X_val, y_val)

        use_cuda = config['cuda']
        use_cuda = use_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, **kwargs)
        self.dataset_train_val[self.source_dataset_name] = {'train': train_loader, 'val':val_loader}

        weights, n_classes = self.get_class_weights()
        class_weights = torch.FloatTensor(weights).to(device)

        criterion = torch.nn.NLLLoss(weight=class_weights)

        if self.source_dataset_name in self.model.keys():
            model = self.model[self.source_dataset_name]
            
        else:
            model = Classifier(X_train.shape[1], LDIM, MODEL_WIDTH, n_classes).to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, threshold=0.05, verbose=True)

        best_val_acc = 0.
        for epoch in range(config['epochs']):
            c, s = 0, 0
            pBar = tqdm(train_loader)
            model.train()
            for sample in pBar:
                x = sample['x'].to(device)
                y = sample['y'].to(device)

                optimizer.zero_grad()
                p = model.predict(x)
                loss = criterion(p, y)
                s = ((s * c) + (float(loss.item()) * len(p))) / (c + len(p))
                c += len(p)
                pBar.set_description('Epoch {} Train: '.format(epoch) + str(round(float(s), 4)))
                loss.backward()
                optimizer.step()
            sch.step(s)

            model.eval()
            y_pred, y_true = [], []
            with torch.no_grad():
                for sample in val_loader:
                    x = sample['x'].to(device)
                    y = sample['y'].to(device)

                    p = model.predict_proba(x)
                    y_pred.append(p.cpu().detach().numpy())
                    y_true.append(y.cpu().detach().numpy())
            y_pred = np.concatenate(y_pred)
            y_true = np.concatenate(y_true)

            val_acc = (y_true == y_pred.argmax(axis=1)).mean()
            print("Validation Accuracy {:.4f}".format(val_acc))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                #torch.save(model.state_dict(), self.path + "/best_first_classifier.pth")
                self.val_stats = {'pred': y_pred, 'true': y_true}
                # Save validation stats to a file
                with open(f'{self.path}/val_stats.json', 'w') as f:
                    json.dump({'pred': y_pred.tolist(), 'true': y_true.tolist()}, f)

        if cmat:
            # Plot validation confusion matrix
            self.plot_cfmt(self.val_stats['pred'], self.val_stats['true'], 0.05, 'val_cfmt.pdf')

        # Finally keep the best model
        #model.load_state_dict(torch.load(self.path + "/best_classifier.pth"))
        self.model[self.source_dataset_name] = model 
        self.model[self.source_dataset_name].eval()

    def get_dataset_train_val(self, data, dataset, config=get_config()['ftune']):
        if dataset in self.dataset_train_val.keys():
            return self.dataset_train_val[dataset]['train'], self.dataset_train_val[dataset]['val']

        use_cuda = config['cuda']
        use_cuda = use_cuda and torch.cuda.is_available()
        kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}

        batch_data = data[data[BATCH] == dataset]
        ind = np.random.randint(0, len(batch_data), len(batch_data))
        gene_mat = batch_data.drop([BATCH, LABELS], axis=1).values[ind]
        labels = np.array([self.class2num[i] for i in batch_data[LABELS][ind]])
        X_train, X_val, y_train, y_val = train_test_split(gene_mat, labels, test_size=config['val_frac'], shuffle=True, random_state=config['seed'])
        train_loader = torch.utils.data.DataLoader(DataLoaderCustom(X_train, y_train), batch_size=config['batch_size'], shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(DataLoaderCustom(X_val, y_val), batch_size=config['batch_size'], shuffle=False, **kwargs)

        self.dataset_train_val[dataset] = {'train': train_loader, 'val':val_loader}
        return self.dataset_train_val[dataset]['train'], self.dataset_train_val[dataset]['val']

    def get_class_weights(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        counts = counts / np.sum(counts)
        weights = 2. / (0.01 + counts) / len(unique)
        return weights, len(unique)

    def get_features(self, gene_mat):
        features = gene_mat.values
        if self.count_normalize:
            features = features / (np.sum(features, axis=1, keepdims=True) + 1e-5) * 1e4
        if self.logt:
            features = np.log(1 + features)
        if self.reduce_method is not None:
            if self.reduce_method == "Var":
                selected_genes = [gene_mat.columns[i] for i in self.variances]
                if selected_genes != self.selected_genes:
                    print("Reorder the genes for the target batch in the same order as the source batch")
                    sys.exit()
                features = features[:, self.variances]
            elif self.reduce_method == "PCA":
                features = self.pca.transform(features)
        if self.scaler is not None:
            self.test_scaler = StandardScaler()
            features = self.test_scaler.fit_transform(features)
        return features

    def get_model(self, dataset_name): # aqui el meollo
        if dataset_name in self.model.keys():
            print("[JindLib] Using custom model for dataset {}".format(dataset_name))
            return self.model[dataset_name]
        else:
            print("[JindLib] No custom model exists for dataset {}. Using {} dataset model".format(dataset_name,
                                                                                                   self.source_dataset_name))
            return self.model[self.source_dataset_name]

    def predict(self, data, config=get_config()['train_classifier'], return_names=False):
        use_cuda = config['cuda']
        use_cuda = use_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print(device)
        kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}

        predictions = []
        for dataset in set(data[BATCH]):
            gene_mat = data[data[BATCH] == dataset].drop(BATCH, axis=1)
            test_dataset = DataLoaderCustom(self.get_features(gene_mat))
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False, **kwargs)

            model = self.get_model(dataset)
            model.eval()

            y_pred, y_true = [], []
            with torch.no_grad():
                for sample in test_loader:
                    x = sample['x'].to(device)
                    p = model.predict_proba(x)
                    y_pred.append(p.cpu().detach().numpy())
            predictions = predictions + y_pred
        predictions = np.concatenate(predictions)

        if return_names:
            preds = np.argmax(y_pred, axis=1)
            predictions = [self.num2class[i] for i in preds]
            predicted_label = pd.DataFrame({"cellname": data.values.index, "predictions": predictions})
            return predicted_label

        return predictions

    def get_encoding(self, data):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
         
        kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}

        if LABELS in data.columns:
            data = data.drop(LABELS, axis=1)

        predictions = []
        for dataset in set(data[BATCH]):
            gene_mat = data[data[BATCH] == dataset].drop(BATCH, axis=1)
            test_dataset = DataLoaderCustom(self.get_features(gene_mat))
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False, **kwargs)

            model = self.get_model(dataset)
            model.eval()

            y_pred, y_true = [], []
            with torch.no_grad():
                for sample in test_loader:
                    x = sample['x'].to(device)
                    p = model.get_repr(x)
                    y_pred.append(p.cpu().detach().numpy())
            predictions = predictions + y_pred
        predictions = np.concatenate(predictions)

        return predictions

    def get_filtered_prediction(self, data, frac=0.05):
        y_pred = self.predict(data)

        if frac != 0:
            preds = self.filter_pred(y_pred, frac)
        else:
            preds = np.argmax(y_pred, axis=1)

        predictions = [self.num2class[i] for i in preds]
        raw_predictions = [self.num2class[i] for i in np.argmax(y_pred, axis=1)]

        dic1 = {"cellname": data.index,
                "raw_predictions": raw_predictions,
                "predictions": predictions}
        dic2 = {self.num2class[i]: list(y_pred[:, i]) for i in range(self.n_classes)}

        dic = {**dic1, **dic2}
        predicted_label = pd.DataFrame(dic)
        predicted_label = predicted_label.set_index("cellname")

        return predicted_label

    def evaluate(self, data, frac=0.05, name=None, return_log=False):

        new_labels = [labels for labels in list(data.labels.unique()) if labels not in self.class2num]
        if new_labels:
            # if there are labels not presented in training we need to create another dictionary that has this info
            true_labels_class2num = self.class2num.copy()
            true_labels_class2num.update({label: max(true_labels_class2num.values())+1 + i for i, label in enumerate(new_labels)})
        else:
            true_labels_class2num = self.class2num # so that the code can be reproducible

        y_pred = self.predict(data.drop(LABELS, axis=1))
        y_true = np.array([true_labels_class2num[i] for i in data[LABELS]])

        if frac != 0:
            preds = self.filter_pred(y_pred, frac)
        else:
            preds = np.argmax(y_pred, axis=1)

        raw_acc = (y_true == np.argmax(y_pred, axis=1)).mean()
        ind = preds != self.n_classes

        # pred_acc = (y_true[ind] == preds[ind]).mean()
        matching_values = (y_true[ind] == preds[ind])
        if np.any(matching_values):
            pred_acc = matching_values.mean()
        else:
            pred_acc = 0
        
        filtered = 1 - np.mean(ind)
        total_count = len(y_true)
        correctly_classified = len(preds[y_true == preds])
        rejected = len(preds[preds == self.n_classes])
        misclassified = total_count - (correctly_classified + rejected)

        arranged_labels = np.arange(0, max(np.max(y_true) + 1, np.max(preds) + 1, self.n_classes + 1))
        cm = confusion_matrix(y_true, preds, labels=arranged_labels)
        
        # remove 'unassigned' from true label rows in cm
        cm = np.delete(cm, (self.n_classes), axis=0)
        # remove extra column in pred label columns
        if new_labels:
            cm = np.delete(cm, range(self.n_classes + 1, self.n_classes + 1 + len(new_labels)), axis=1)
        # if cm.shape[1] > (self.n_classes + 1): # version akash
        #     cm = np.delete(cm, (self.n_classes + 1), axis=1)

        # The code calculates the Average Precision (AP) values using the compute_ap function. This function iterates through each class 
        # in y_pred and calculates the corresponding AP using the calc_pr function. The AP values are stored in the aps array.
            
        # aps = np.zeros((len(cm), 1))
        aps = np.zeros((cm.shape[1]-1, 1)) # can be cell-types in the cm not presented in the model training. Remove these from the calculations
        aps[:self.n_classes] = np.array(compute_ap(y_true, y_pred)).reshape(-1, 1)
        mAP = np.true_divide(aps.sum(), (aps != 0).sum()) # mean average precision 

        if self.config['cmat_print_counts']:
            values_format = '.0f'
            total_actuals = np.zeros((cm.shape[0], 1), dtype=int)
            total_actuals[:cm.shape[0]] = cm.sum(axis=1, keepdims=True)
            cm = np.concatenate([cm, total_actuals], axis=1)
            total_predictions = cm.sum(axis=0, keepdims=True)
            cm = np.concatenate([cm, total_predictions])
            # cm[cm.shape[0] - 1][cm.shape[1] - 1] = 0 ??? why, makes no sense
            class_true_labels = [key for key in true_labels_class2num.keys() if key != 'Unassigned'] + ['Total Counts']
            class_pred_labels = list(self.class2num.keys()) + ['Total Counts']
        else:
            values_format = '.2g'
            cm = np.concatenate([normalize(cm, normalize='true'), aps], axis=1)
            class_true_labels = [key for key in true_labels_class2num.keys() if key != 'Unassigned'] + ['Novel'] + ['AP']
            class_pred_labels = list(self.class2num.keys()) + ['Novel'] + ['AP']

        eval_result = 'T {} #Rej {} ({:.1f}%) corrct {} (raw {:.3f}% eff {:.3f}%) incorrct {} mAP {:.3f}% '.format(total_count,
                rejected, filtered*100, correctly_classified, raw_acc*100, pred_acc*100, misclassified, mAP*100)
        print('[JindLib][Evaluate] Accuracy: {} {}'.format(eval_result, set(data[BATCH])))

        if name is not None:
            cm_ob = ConfusionMatrixPlot(cm, class_true_labels, class_pred_labels, title='{} {}'.format(name, eval_result))
            factor = max(1, len(cm) // 10)
            fig = plt.figure(figsize=(10 * factor, 8 * factor))
            cm_ob.plot(values_format=values_format, ax=fig.gca())
            plt.tight_layout()
            plt.savefig('{}/{}'.format(self.path, name))
            self.plot_embeddings[name] = self.plot_embeddings.get(name, {})
            self.plot_embeddings[name]['matrix'] = cm_ob
            self.conf_matrix.append((name, cm_ob))

        predictions = [self.num2class[i] for i in preds]
        raw_predictions = [self.num2class[i] for i in np.argmax(y_pred, axis=1)]

        dic1 = {"cellname": data.index,
                "raw_predictions": raw_predictions,
                "predictions": predictions,
                "labels": data[LABELS]}
        dic2 = {self.num2class[i]: list(y_pred[:, i]) for i in range(self.n_classes)}
        dic = {**dic1, **dic2}
        predicted_label = pd.DataFrame(dic)
        predicted_label = predicted_label.set_index("cellname")

        if return_log:
            return predicted_label
        return predicted_label

    def plot_cfmt(self, y_pred, y_true, frac=0.05, name=None):
        if frac != 0:
            preds = self.filter_pred(y_pred, frac)
        else:
            preds = np.argmax(y_pred, axis=1)
        pretest_acc = (y_true == np.argmax(y_pred, axis=1)).mean()
        test_acc = (y_true == preds).mean()
        ind = preds != self.n_classes
        pred_acc = (y_true[ind] == preds[ind]).mean()
        filtered = 1 - np.mean(ind)

        if name is not None:
            arranged_labels = np.arange(0, max(np.max(y_true) + 1, np.max(preds) + 1, self.n_classes + 1))
            cm = normalize(confusion_matrix(y_true, preds, labels=arranged_labels), normalize='true')
            cm = np.delete(cm, (self.n_classes), axis=0)
            if cm.shape[1] > (self.n_classes + 1):
                cm = np.delete(cm, (self.n_classes + 1), axis=1)
            aps = np.zeros((len(cm), 1))
            aps[:self.n_classes] = np.array(compute_ap(y_true, y_pred)).reshape(-1, 1)
            cm = np.concatenate([cm, aps], axis=1)

            class_labels = list(self.class2num.keys()) + ['Novel'] + ['AP']
            APs = aps[:self.n_classes]
            mAP = np.true_divide(aps.sum(), (aps != 0).sum())
            title = 'Accuracy Raw {:.3f} Eff {:.3f} Rej {:.3f} mAP {:.3f}'.format(pretest_acc, pred_acc, filtered, mAP)
            cm_ob = ConfusionMatrixPlot(cm, class_labels, class_labels, title=title)
            factor = max(1, len(cm) // 10)
            fig = plt.figure(figsize=(10 * factor, 8 * factor))
            cm_ob.plot(values_format='0.2f', ax=fig.gca())

            plt.tight_layout()
            plt.savefig('{}/{}'.format(self.path, name))
            plt.close()

    def get_thresholds(self, outlier_frac):
        thresholds = 0.9 * np.ones((self.n_classes))
        probs_train = self.val_stats['pred']
     
        for top_klass in range(self.n_classes):
            ind = (np.argmax(probs_train, axis=1) == top_klass)  # & (y_train == top_klass)

            if np.sum(ind) != 0:
                best_prob = np.max(probs_train[ind], axis=1)
                best_prob = np.sort(best_prob)
                l = int(outlier_frac * len(best_prob)) + 1
           
                if l < (len(best_prob)):
                    thresholds[top_klass] = best_prob[l]
        return thresholds

    def filter_pred(self, pred, outlier_frac):
        thresholds = self.get_thresholds(outlier_frac)

        pred_class = np.argmax(pred, axis=1)
        prob_max = np.max(pred, axis=1)

        ind = prob_max < thresholds[pred_class]
        pred_class[ind] = self.n_classes  # assign unassigned class
        return pred_class

    def get_TSNE(self, features):
        pca = PCA(n_components=50)
        reduced_feats = pca.fit_transform(features)
        embeddings = TSNE(n_components=2, verbose=1, n_jobs=-1, perplexity=50, random_state=43).fit_transform(
            reduced_feats)
        return embeddings

    def plot_tsne_of_batches(self, data, plot_name):
        if not self.config['plot_tsne']:
            return

        print("\n[JindLib][plot_TSNE_of_batches] Plotting TSNE for ", plot_name, set(data[BATCH]))
        emb = self.get_TSNE(self.get_encoding(data))
        df = pd.DataFrame({'tSNE_x': emb[:, 0], 'tSNE_y': emb[:, 1], 'Labels': data[LABELS], 'Batch': data[BATCH]})
        plot_and_save_tsne(df, self.path, plot_name)
        self.plot_embeddings[plot_name] = self.plot_embeddings.get(plot_name, {})
        self.plot_embeddings[plot_name]['tsne'] = df

    def remove_effect(self, train_data, test_data, config):
        if LABELS in train_data.columns:
            train_data = train_data.drop(LABELS, axis=1)
        if LABELS in test_data.columns:
            test_data = test_data.drop(LABELS, axis=1)

        train_datasets = train_data[BATCH]
        test_dataset_name = test_data[BATCH][0]
        features_train_data = self.get_features(train_data.drop(BATCH, axis=1))
        features_test_data = self.get_features(test_data.drop(BATCH, axis=1))

        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        use_cuda = config['cuda'] and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
      
        kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}

        batch2_loader = torch.utils.data.DataLoader(DataLoaderCustom(features_test_data),
                                                    batch_size=config['batch_size'], shuffle=False, **kwargs)

        train_models = {self.source_dataset_name: self.model[self.source_dataset_name].to(device)}
     
        for param in train_models[self.source_dataset_name].parameters():
            param.requires_grad = False
        # Define new model and intialize it with the same parameter values as trained model
        model_copy = Classifier(features_train_data.shape[1], LDIM, MODEL_WIDTH, self.n_classes).to(device)
        model_copy.load_state_dict(train_models[self.source_dataset_name].state_dict())
        for param in model_copy.parameters():
            param.requires_grad = False
        model_for_test_dataset = ClassifierBig(model_copy, features_train_data.shape[1], LDIM, GLDIM).to(device)

        # Load models of all train datasets and make them constant 
        for dataset in set(train_data[BATCH]):
            train_models[dataset] = self.get_model(dataset)
     
            for param in train_models[dataset].parameters():
                param.requires_grad = False
            train_models[dataset].eval() # hasta aquí está el meollo

        disc = Discriminator(LDIM).to(device)

        G_decay = config.get("gdecay", 1e-2)
        D_decay = config.get("ddecay", 1e-6)
        max_count = config.get("maxcount", 3)
        sigma = config.get("sigma", 0.0)

        optimizer_G = torch.optim.RMSprop(model_for_test_dataset.parameters(), lr=1e-4, weight_decay=G_decay)
        optimizer_D = torch.optim.RMSprop(disc.parameters(), lr=1e-4, weight_decay=D_decay)
        adversarial_weight = torch.nn.BCELoss(reduction='none')
        adversarial_loss = torch.nn.BCELoss()
        sample_loss = torch.nn.BCELoss()

        Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        bs = min(config['batch_size'], len(features_test_data), len(features_train_data))
        count = 0
        dry_epochs = 0
        best_rej_frac = 1.0

        # Evaluate the initialized model (avoid saving a worse model later with a higher rejection rate)
        model_for_test_dataset.eval()
        self.model[test_dataset_name] = model_for_test_dataset
        predictions = self.get_filtered_prediction(test_data, frac=0.05)

        rej_frac = np.mean(predictions["predictions"] == "Unassigned")
        if rej_frac < best_rej_frac:
            print(f"[JindLib] Updated Rejected cells from {best_rej_frac:.3f} to {rej_frac:.3f}")
            best_rej_frac = rej_frac
            torch.save(model_for_test_dataset.state_dict(), self.path + "/target_{}_bestbr.pth".format(test_dataset_name))

        for epoch in range(config['epochs']):
            if len(batch2_loader) < 50:
                pBar = tqdm(range(40))
            else:
                pBar = tqdm(batch2_loader)
            model_for_test_dataset.eval()
            disc.train()
            c1, s1 = 0, 0.7
            c2, s2 = 0, 0.7
            for sample in pBar:
                valid = Variable(Tensor(bs, 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(bs, 1).fill_(0.0), requires_grad=False)
                sample_loss = torch.nn.BCELoss()
                disc.eval()

                for i in range(1):
                    ind = np.random.randint(0, (len(features_test_data)), bs)
                    batch2_inps = Variable(torch.from_numpy(features_test_data[ind])).to(device).type(Tensor)
                    optimizer_D.zero_grad()
                    optimizer_G.zero_grad()

                    batch2_code = model_for_test_dataset.get_repr(batch2_inps)
                    g_loss = sample_loss(disc(batch2_code + sigma * torch.randn(batch2_inps.shape[0], LDIM).to(device)),
                                         valid)  # + 0.001 * penalty
                    if s2 > 0.4:
                        g_loss.backward()
                        optimizer_G.step()
                    s2 = ((s2 * c2) + (float(g_loss.item()) * len(batch2_code))) / (c2 + len(batch2_code))
                    c2 += len(batch2_code)

                    if s2 == 0 or g_loss.item() == 0:
                        model_for_test_dataset.reinitialize()
                        count = 0  # reset count as well
                        dry_epochs = 0

                sample_loss = torch.nn.BCELoss()
                model_for_test_dataset.eval()
                disc.train()
                for i in range(2):
                    if i != 0:
                        ind = np.random.randint(0, (len(features_test_data)), bs)
                        batch2_inps = Variable(torch.from_numpy(features_test_data[ind])).to(device).type(Tensor)
                        batch2_code = model_for_test_dataset.get_repr(batch2_inps)
                    optimizer_D.zero_grad()
                    ind = np.random.randint(0, (len(features_train_data)), bs)
                    batch1_datasets = train_datasets[ind]
                    batch1_code = torch.empty((0, LDIM))
                    for dataset in set(batch1_datasets):
                        dataset_ind = [i for i in ind if train_datasets[i] == dataset]
                        batch1_inps = Variable(torch.from_numpy(features_train_data[dataset_ind])).to(device).type(Tensor)
                        batch1_code = batch1_code.to(device)  # Mover a dispositivo
                        batch1_code = torch.cat((batch1_code, train_models[dataset].get_repr(batch1_inps)))

                    real_loss = sample_loss(
                        disc(batch1_code + sigma * torch.randn(batch1_code.shape[0], LDIM).to(device)),
                        valid[:batch1_code.size()[0]])

                    fake_loss = sample_loss(
                        disc(batch2_code.detach() + sigma * torch.randn(batch2_code.shape[0], LDIM).to(device)), fake)
                    d_loss = 0.5 * (real_loss + fake_loss)

                    if s2 < 0.8 or s1 > 1.0:
                        d_loss.backward()
                        optimizer_D.step()
                    s1 = ((s1 * c1) + (float(d_loss.item()) * len(batch1_code))) / (c1 + len(batch1_code))
                    c1 += len(batch1_code)

                    if s1 == 0 or d_loss.item() == 0:
                        model_for_test_dataset.reinitialize()
                        count = 0
                        dry_epochs = 0

                pBar.set_description('Epoch {} G Loss: {:.3f} D Loss: {:.3f}'.format(epoch, s2, s1))

            if (s2 < 0.78) and (s2 > 0.5) and (s1 < 0.78) and (s1 > 0.5):
                count += 1
                self.model[test_dataset_name] = model_for_test_dataset
                predictions = self.get_filtered_prediction(test_data, frac=0.05)

                rej_frac = np.mean(predictions["predictions"] == "Unassigned")
                if rej_frac < best_rej_frac:
                    print(f"Updated Rejected cells from {best_rej_frac:.3f} to {rej_frac:.3f}")
                    best_rej_frac = rej_frac
                    torch.save(model_for_test_dataset.state_dict(),
                               self.path + "/target_{}_bestbr.pth".format(test_dataset_name))

                dry_epochs = 0
                if count >= max_count:
                    break
            else:
                dry_epochs += 1
                if dry_epochs == max_count:
                    print("Loss not improving, stopping alignment")
                    break

        #if not os.path.isfile(self.path + "/target_{}_bestbr".format(test_dataset_name)):
            #print("Warning: Alignment did not succeed properly, try changing the gdecay or ddecay!")
            #torch.save(model_for_test_dataset.state_dict(), self.path + "/target_{}_bestbr.pth".format(test_dataset_name))

        #model_for_test_dataset.load_state_dict(torch.load(self.path + "/target_{}_bestbr.pth".format(test_dataset_name)))
        self.model = self.update_model_copies(self.model, model_for_test_dataset, test_dataset_name)

    def domain_adapt(self, data, config):
        test_dataset_name = data[BATCH][0]

        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        use_cuda = config['cuda'] and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
    
        weights, n_classes = self.get_class_weights()
        class_weights = torch.FloatTensor(weights).to(device)
        criterion = torch.nn.NLLLoss(weight=class_weights)

        train_loader, val_loader = self.get_dataset_train_val(data, test_dataset_name, config)

        model = ClassifierBig(self.model[self.source_dataset_name], data.shape[1]-2, LDIM, GLDIM).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, threshold=0.05, verbose=True)
        for param in model.parameters():
            param.requires_grad = True
        for param in model.m1.fc.parameters():
            param.requires_grad = False
        for param in model.m1.fc1.parameters():
            param.requires_grad = False

        best_loss = 100.
        for epoch in range(config['epochs_da']):
            c, s = 0, 0
            pBar = tqdm(train_loader)
            model.train()
            for sample in pBar:
                x = sample['x'].to(device)
                y = sample['y'].to(device)
                optimizer.zero_grad()
                p = model.predict(x)
                loss = criterion(p, y)
                s = ((s * c) + (float(loss.item()) * len(p))) / (c + len(p))
                c += len(p)
                pBar.set_description('Epoch {}: {}: {}:'.format(epoch, str(round(float(s), 4)), test_dataset_name))
                loss.backward()
                optimizer.step()
            sch.step(s)

            model.eval()
            y_pred, y_true = [], []
            with torch.no_grad():
                for sample in val_loader:
                    x = sample['x'].to(device)
                    y = sample['y'].to(device)
                    p = model.predict_proba(x)
                    y_pred.append(p.cpu().detach().numpy())
                    y_true.append(y.cpu().detach().numpy())
            y_pred = np.concatenate(y_pred)
            y_true = np.concatenate(y_true)

            val_acc = (y_true == y_pred.argmax(axis=1)).mean()
            print("Validation Accuracy = {:.4f}. Loss = {:.4s}".format(val_acc, str(round(float(s), 4))))
            if s <= best_loss:
                best_loss = s
                #torch.save(model.state_dict(), self.path + "/{}_bestbr.pth".format(test_dataset_name))
                self.model[test_dataset_name] = model
                self.evaluate(data)

        # Finally keep the best model
        #model.load_state_dict(torch.load(self.path + "/{}_bestbr.pth".format(test_dataset_name)))
        self.model[test_dataset_name] = model

    def clone_models(self, data):
        device = torch.device("cpu")
        base_model = Classifier(data[data[BATCH] == self.source_dataset_name].shape[1]-2, LDIM, MODEL_WIDTH, self.n_classes).to(device)
        model_copies = {self.source_dataset_name: base_model}
        for dataset in [d for d in self.model.keys() if d!=self.source_dataset_name]:
            model_copy = ClassifierBig(base_model, data[data[BATCH] == dataset].shape[1]-2, LDIM, GLDIM).to(device)
            model_copy.load_state_dict(self.model[dataset].to(device).state_dict())
            model_copies[dataset] = model_copy
        return model_copies

    def ftune_demised(self, test_data, config, cmat=True, retrain_encoder=False):
        test_data = self.parse_test_data_for_ftune(test_data)
        test_dataset_name = test_data[BATCH][0]

        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        use_cuda = config['cuda']
        use_cuda = use_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
    
        train_loader, val_loader = self.get_dataset_train_val(test_data, test_dataset_name, config)

        weights, n_classes = self.get_class_weights()
        class_weights = torch.FloatTensor(weights).to(device)
        criterion = torch.nn.NLLLoss(weight=class_weights)

        model = self.model[test_dataset_name]
        for param in model.parameters():
            param.requires_grad = False
        for param in model.m1.fc.parameters():
            param.requires_grad = True
        for param in model.m1.fc1.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, threshold=0.05, verbose=True)

        best_val_acc = 0.
        for epoch in range(config['epochs']):
            c, s = 0, 0
            pBar = tqdm(train_loader)
            model.train()
            for sample in pBar:
                x = sample['x'].to(device)
                y = sample['y'].to(device)
                optimizer.zero_grad()
                p = model.predict(x)
                loss = criterion(p, y)
                s = ((s * c) + (float(loss.item()) * len(p))) / (c + len(p))
                c += len(p)
                pBar.set_description('Epoch {}: {}: {}'.format(epoch, str(round(float(s), 4)), test_dataset_name))
                loss.backward()
                optimizer.step()
            sch.step(s)

            model.eval()
            y_pred, y_true = [], []
            with torch.no_grad():
                for sample in val_loader:
                    x = sample['x'].to(device)
                    y = sample['y'].to(device)
                    p = model.predict_proba(x)
                    y_pred.append(p.cpu().detach().numpy())
                    y_true.append(y.cpu().detach().numpy())
            y_pred = np.concatenate(y_pred)
            y_true = np.concatenate(y_true)

            # if s < 0.0001 and epoch>5:
            #     break
            val_acc = (y_true == y_pred.argmax(axis=1)).mean()
            print("Validation Accuracy {:.4f}".format(val_acc))
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), self.path + "/{}_bestbr_ftune.pth".format(test_dataset_name))
                val_stats = {'pred': y_pred, 'true': y_true}
                self.model[test_dataset_name] = model
                self.evaluate(test_data)

        if cmat:
            self.plot_cfmt(val_stats['pred'], val_stats['true'], 0.05, 'val_cfmtftune.pdf')

        # Finally keep the best model
        model.load_state_dict(torch.load(self.path + "/{}_bestbr_ftune.pth".format(test_dataset_name)))
        self.model[test_dataset_name] = model

    def stamp_labels_for_test_data(self, test_data):
        features = self.get_features(test_data.drop(BATCH, axis=1))
        y_pred = self.predict(test_data)
        preds = self.filter_pred(y_pred, 0.1)

        ind = preds != self.n_classes
        test_dataset_name = test_data[BATCH][0]
        test_data = pd.DataFrame(features[ind], index=test_data[ind].index, columns=test_data.drop(BATCH, axis=1).columns)
        test_data[LABELS] = [self.num2class[i] for i in preds[ind]]
        test_data[BATCH] = [test_dataset_name]*len(test_data)
        print("[JindLib] Using {} high confidence cells out of {} for fine tuning".format(len(test_data), len(ind)))
        return test_data

    def ftune(self, data, config, datasets_to_train=None, cmat=True):  
        datasets_to_train = list(set(data[BATCH])) if datasets_to_train is None else datasets_to_train
        test_dataset_name = datasets_to_train[len(datasets_to_train)-1]
         
        for i in datasets_to_train:
            batch_data = data[data[BATCH] == i]
             
        metric = config['metric']
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        torch.backends.cudnn.deterministic = True

        use_cuda = config['cuda']
        use_cuda = use_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
     
        weights, n_classes = self.get_class_weights()
        class_weights = torch.FloatTensor(weights).to(device)
        criterion = torch.nn.NLLLoss(weight=class_weights)

        dataset_optimizer = {}
        for dataset in datasets_to_train:
            optimizer = optim.Adam(self.model[dataset].parameters(), lr=1e-4)
            sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, threshold=0.05, verbose=True)
            dataset_optimizer[dataset] = {'sch': sch, 'optimizer': optimizer}

        model_copies = self.model
        best_loss = 100000.
        best_val_acc = 0.
        for epoch in range(config['epochs']):
            epoch_val_acc = {}
            loss = 0
            for dataset in datasets_to_train:
                c, s = 0, 0

                train_loader, val_loader = self.get_dataset_train_val(data, dataset, config)
                optimizer = dataset_optimizer[dataset]['optimizer']
                sch = dataset_optimizer[dataset]['sch']
                pBar = tqdm(train_loader)
                model = model_copies[dataset]

                if dataset == self.source_dataset_name:
                    for param in model.parameters():
                        param.requires_grad = True
                else:
                    for param in model.parameters():
                        param.requires_grad = False
                    for param in model.m1.parameters():
                        param.requires_grad = True

                model.train()
                for sample in pBar:
                    x = sample['x'].to(device)
                    y = sample['y'].to(device)
                    optimizer.zero_grad()
                    p = model.predict(x)
                    loss = criterion(p, y)
                    s = ((s * c) + (float(loss.item()) * len(p))) / (c + len(p))
                    c += len(p)
                    pBar.set_description('Epoch {}: Loss {:4s}: Dataset {:<{}s}:'.format(epoch, str(round(float(s), 4)), dataset, len(max(datasets_to_train, key=len))))
                    loss.backward()
                    optimizer.step()
                sch.step(s)
                loss += s

                model.eval()
                y_pred, y_true = [], []
                with torch.no_grad():
                    for sample in val_loader:
                        x = sample['x'].to(device)
                        y = sample['y'].to(device)
                        p = model.predict_proba(x)
                        y_pred.append(p.cpu().detach().numpy())
                        y_true.append(y.cpu().detach().numpy())
                y_pred = np.concatenate(y_pred)
                y_true = np.concatenate(y_true)

                val_acc = (y_true == y_pred.argmax(axis=1)).mean()
                model_copies = self.update_model_copies(model_copies, model, dataset, print_log=False)
                dataset_optimizer[dataset] = {'sch': sch, 'optimizer': optimizer}
                epoch_val_acc[dataset] = val_acc

            epoch_val_acc_sum = sum(epoch_val_acc.values())/len(datasets_to_train)
            epoch_val_acc = {dataset: round(epoch_val_acc[dataset], 4) for dataset in epoch_val_acc}
            print("Validation accuracy = {:.4f}. Loss = {:4s}".format(epoch_val_acc_sum, str(round(float(loss), 4))), epoch_val_acc)

            if (metric=='loss' and loss <= best_loss) or (metric=='accuracy' and epoch_val_acc_sum>=best_val_acc):
                best_loss = loss
                best_val_acc = epoch_val_acc_sum
                #torch.save(model_copies[test_dataset_name].state_dict(), self.path + "/{}_bestbr_ftune.pth".format(test_dataset_name))

                # Updata val_stats
                self.val_stats = {'pred': y_pred, 'true': y_true}
                
                # Save validation stats to a file
                with open(f'{self.path}/val_stats.json', 'w') as f:
                    json.dump({'pred': y_pred.tolist(), 'true': y_true.tolist()}, f)
                if cmat:
                # Plot validation confusion matrix
                    self.plot_cfmt(self.val_stats['pred'], self.val_stats['true'], 0.05, f'val_cfmtftune_{test_dataset_name}.pdf')
                 
                self.model = self.update_model_copies(self.model, model_copies[test_dataset_name], test_dataset_name)
                for dataset_evaluate in set(data[BATCH]):
                    self.evaluate(data[data[BATCH] == dataset_evaluate])

        # Finally keep the best model
        #model_copies[test_dataset_name].load_state_dict(torch.load(self.path + "/{}_bestbr_ftune.pth".format(test_dataset_name)))
        self.model = self.update_model_copies(self.model, model_copies[test_dataset_name], test_dataset_name)

    def compare_models(self, model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                # if key_item_1[0] == key_item_2[0]:
                #     print('Mismatch found at', key_item_1[0])
                return "not equals"
        if models_differ == 0:
            return "=="

    def update_model_copies(self, models, model_to_copy, model_to_copy_name, print_log=True):
        s = "[JindLib][Encoder_Classifier_Compare]"

        for model_to_update_name in models.keys():
            #print("[JindLib] Updating model {} with {}".format(model_to_update_name, model_to_copy_name)) if print_log else ''
            if model_to_update_name == self.source_dataset_name:
                if model_to_copy_name == self.source_dataset_name:
                    #print("{} {} {} {}".format(s, model_to_copy_name, self.compare_models(model_to_copy, models[model_to_update_name]), model_to_update_name))
                    models[model_to_update_name].fc.load_state_dict(model_to_copy.fc.state_dict())
                    models[model_to_update_name].fc1.load_state_dict(model_to_copy.fc1.state_dict())
                    #print("{} {} {} {}".format(s, model_to_copy_name, self.compare_models(model_to_copy, models[model_to_update_name]), model_to_update_name))
                else:
                    #print("{} {} {} {}".format(s, model_to_copy_name, self.compare_models(model_to_copy.m1, models[model_to_update_name]), model_to_update_name))
                    models[model_to_update_name].fc.load_state_dict(model_to_copy.m1.fc.state_dict())
                    models[model_to_update_name].fc1.load_state_dict(model_to_copy.m1.fc1.state_dict())
                    #print("{} {} {} {}".format(s, model_to_copy_name, self.compare_models(model_to_copy.m1, models[model_to_update_name]), model_to_update_name))
            else:
                if model_to_copy_name == self.source_dataset_name:
                    #print("{} {} {} {}".format(s, model_to_copy_name, self.compare_models(model_to_copy, models[model_to_update_name].m1), model_to_update_name))
                    models[model_to_update_name].m1.fc.load_state_dict(model_to_copy.fc.state_dict())
                    models[model_to_update_name].m1.fc1.load_state_dict(model_to_copy.fc1.state_dict())
                    #print("{} {} {} {}".format(s, model_to_copy_name, self.compare_models(model_to_copy, models[model_to_update_name].m1), model_to_update_name))
                else:
                    #print("{} {} {} {}".format(s, model_to_copy_name, self.compare_models(model_to_copy.m1, models[model_to_update_name].m1), model_to_update_name))
                    models[model_to_update_name].m1.fc.load_state_dict(model_to_copy.m1.fc.state_dict())
                    models[model_to_update_name].m1.fc1.load_state_dict(model_to_copy.m1.fc1.state_dict())
                    #print("{} {} {} {}".format(s, model_to_copy_name,self.compare_models(model_to_copy.m1, models[model_to_update_name].m1),model_to_update_name))
        return models

    def to_pickle(self, name):
        self.raw_features = None
        self.reduced_features = None
        with open('{}/{}'.format(self.path, name), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)