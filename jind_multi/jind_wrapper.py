from datetime import datetime
from .config_loader import get_config
from .jindlib import JindLib
from .utils import plot_cmat_timeline, remove_pth_files
import re
import pandas as pd
import os
import torch
import json

global BATCH, LABELS
BATCH = 'batch'
LABELS = 'labels'

def evaluate(jind_obj, data, name_tag=None, test_data_name=None):
    source = jind_obj.source_dataset_name
    datasets = ([source] if source in set(data[BATCH]) else []) + ([d for d in set(data[BATCH]) if d!=source and d!=test_data_name]) + ([test_data_name] if test_data_name is not None else [])
    
    for dataset in datasets:
        jind_obj.evaluate(data[data[BATCH] == dataset], name="{}_{}.pdf".format(dataset, name_tag) if name_tag is not None else None)

def perform_domain_adaptation(jind_obj, train_data, test_data, config):
    jind_obj.evaluate(test_data, name="{}_{}.pdf".format(test_data[BATCH][0], "initial"))

    print("[JindWrapper] Performing Domain Adaptation for {} dataset".format(set(test_data[BATCH])))
    jind_obj.domain_adapt(test_data, config['GAN'])
    jind_obj.evaluate(test_data, name="{}_{}.pdf".format(test_data[BATCH][0], "adapt"))

    if config['ftune_intermediate']:
        print("\n[JindWrapper] Fine tuning in parallel for {} dataset".format(set(train_data.append(test_data)[BATCH])))
        config['ftune']['metric'] = 'loss'
        jind_obj.ftune(train_data.append(test_data), config['ftune'], datasets_to_train=list(train_data.append(test_data)[BATCH].unique())) 
        evaluate(jind_obj, train_data.append(test_data), name_tag="ftune", test_data_name=test_data[BATCH][0])

def perform_jind_training(jind_obj, train_data, test_data, config):
    train_dataset_names = set(train_data[BATCH])
    test_dataset_name = set(test_data[BATCH])

    base_plot_name = "train{}-test{}".format(train_dataset_names, test_dataset_name)
    plot_name_initial = "{}_initial.pdf".format(base_plot_name)
    plot_name_after_GAN = "{}_GAN.pdf".format(base_plot_name)
    plot_name_after_fineTune = "{}_ftune.pdf".format(base_plot_name)
    plot_name_prediction_results = "analyze_predictions"

    if test_data[LABELS].nunique() > 1:
        jind_obj.evaluate(test_data, name="{}_{}.pdf".format(test_data[BATCH][0], "initial")) # in GAN training we want to evaluate just the TARGET DATA
    jind_obj.plot_tsne_of_batches(train_data.append(test_data), plot_name_initial) 

    print("[JindWrapper] Removing batch effect by mapping {} onto {}".format(test_dataset_name, train_dataset_names))
    jind_obj.remove_effect(train_data, test_data, config['GAN'])
    if test_data[LABELS].nunique() > 1:
        jind_obj.evaluate(test_data, name="{}_{}.pdf".format(test_data[BATCH][0], "GAN"))
    jind_obj.plot_tsne_of_batches(train_data.append(test_data), plot_name_after_GAN)

    print("[JindWrapper] Fine tuning JIND for {} dataset".format(test_dataset_name))
    test_data_without_original_labels = test_data.drop(LABELS, axis=1)
    jind_obj.ftune(jind_obj.stamp_labels_for_test_data(test_data_without_original_labels), config['ftune'])
    if test_data[LABELS].nunique() > 1:
        jind_obj.evaluate(test_data, name="{}_{}.pdf".format(test_data[BATCH][0], "ftune"))
    jind_obj.plot_tsne_of_batches(train_data.append(test_data), plot_name_after_fineTune)

    print("[JindWrapper] Getting the test labels predictions")
    predicted_label = jind_obj.get_filtered_prediction(test_data_without_original_labels)
    
    # Adding the original labels back to the predictions
    if test_data['labels'].nunique() > 1:
        predicted_label['labels'] = test_data[LABELS].values
        # Analyze results in test data if we have the labels (if tsne = true in config)
        jind_obj.plot_tsne_of_batches(test_data, plot_name_prediction_results, predicted_label)

    return predicted_label

def run_multi_mode(data, train_dataset_names, target_dataset_name, path, config={}, source_dataset_name=None):
    print("[JindWrapper] MULTI MODE. source = {}. target = {}".format(train_dataset_names, target_dataset_name))
    
    train_data = data[data[BATCH].isin(train_dataset_names)]
    jind = JindWrapper(train_data, path, config=config, train_dataset_names=train_dataset_names, source_dataset_name=source_dataset_name)
    raw_acc_per, eff_acc_per, mAP_per, rejected_per = jind.train(data[data[BATCH]==target_dataset_name])
    # save_results_to_sheets(jind, target_dataset_name, 'Multi', config)
    return jind, raw_acc_per, eff_acc_per, mAP_per, rejected_per

def run_combined_mode(data, source_dataset_names, target_dataset_name, path, config=get_config()):
    source_dataset_names = [source_dataset_names] if len(source_dataset_names)==0 else source_dataset_names
    
    print("[JindWrapper] COMBINED MODE. source = {}. target = {}".format(source_dataset_names, target_dataset_name))
    train_data = data[data[BATCH].isin(source_dataset_names)].copy()
    # train_data[BATCH] = ['|'.join(source_dataset_names)] * len(train_data)
    train_data.loc[:, BATCH] = '|'.join(source_dataset_names)
    jind = JindWrapper(train_data, path, config=config)
    raw_acc_per, eff_acc_per, mAP_per, rejected_per = jind.train(data[data[BATCH]==target_dataset_name])
    # save_results_to_sheets(jind, target_dataset_name, 'Combine')
    return jind, raw_acc_per, eff_acc_per, mAP_per, rejected_per

def run_single_mode(data, source_dataset_name, target_dataset_name, path, config=get_config()):
    print("[JindWrapper] SINGLE MODE. source = {}. target = {}".format(source_dataset_name, target_dataset_name))
    jind = JindWrapper(data[data[BATCH]==source_dataset_name], path, config=config)
    raw_acc_per, eff_acc_per, mAP_per, rejected_per = jind.train(data[data[BATCH]==target_dataset_name])
    # save_results_to_sheets(jind, target_dataset_name, 'Single')
    return jind, raw_acc_per, eff_acc_per, mAP_per, rejected_per


class JindWrapper:
    def __init__(self, train_data, output_path, train_dataset_names=None, config=get_config(), source_dataset_name=None):
        self.train_dataset_names = train_dataset_names if train_dataset_names is not None else list(set(train_data[BATCH]))
        if source_dataset_name and source_dataset_name not in self.train_dataset_names:
            self.train_dataset_names.insert(0, source_dataset_name)
        self.path = output_path
        self.config = get_config(user_conf=config)
        self.train_data = train_data[train_data[BATCH].isin(self.train_dataset_names)]
        self.batches_trained = []
        self.source_dataset_name = source_dataset_name

    def train(self, target_data, model=None, val_stats=None):
        print("\n[JindWrapper] Starting JIND training. Run Id = {}".format(self.path.split('/').pop()))
        self.target_data = target_data
        self.source_dataset_name = self.get_dataset_with_least_batch_effect() if self.source_dataset_name is None else self.source_dataset_name
        self.intermediate_dataset_names = [n for n in self.train_dataset_names if n != self.source_dataset_name]
        
        # Initialize JindLib object 
        if model is not None:
            self.jind_obj = JindLib(self.train_data[self.train_data[BATCH] == self.source_dataset_name], self.path, self.config, model=model, val_stats=val_stats)
            print("\n[JindWrapper] An already trained model with its val_stats vas provided")
        else:
            self.jind_obj = JindLib(self.train_data[self.train_data[BATCH] == self.source_dataset_name], self.path, self.config)
            print("\n[JindWrapper] No trained model was provided")
            print("\n[JindWrapper] Training encoder classifier using {} dataset".format(self.source_dataset_name), self.train_data[self.train_data[BATCH] == self.source_dataset_name].shape)
        
        # Train classifier only if a model is not provided
        if model is None:
            print("\n[JindWrapper] Training classifier ...")
            self.jind_obj.train_classifier(config=self.config['train_classifier'])
            self.jind_obj.evaluate(self.train_data[self.train_data[BATCH] == self.source_dataset_name], name="{}_{}.pdf".format(self.source_dataset_name, "after-train_classifier" ))
            self.batches_trained = [self.source_dataset_name]

            # Plot t-sne
            base_plot_name = "train{}-test{}".format(self.train_dataset_names, set(self.target_data[BATCH]))  
            plot_name_initial = "{}_after_train_classifier.pdf".format(base_plot_name)  
            self.jind_obj.plot_tsne_of_batches(self.train_data.append(self.target_data), plot_name_initial) 

            # Train Intermediate datasets
            for train_dataset_name in self.intermediate_dataset_names:
                train_data = self.train_data[self.train_data[BATCH].isin(self.batches_trained)]
                test_data = self.train_data[self.train_data[BATCH] == train_dataset_name]

                print("\n[JindWrapper] Training JIND for intermediate dataset {}".format(train_dataset_name), "Train data shape = ", train_data.shape, "Intermediate data shape = ", test_data.shape)
                perform_domain_adaptation(self.jind_obj, train_data, test_data, self.config)
                self.batches_trained = self.batches_trained + [train_dataset_name]

            train_data = self.train_data[self.train_data[BATCH].isin(self.batches_trained)]

            # Re-training all JIND labeled dataset models in parallel
            if len(self.intermediate_dataset_names):
                if self.config['retrain_intermediate']:
                    print("\n[JindWrapper] Re-training all JIND labeled dataset models in parallel - ", set(train_data[BATCH]))
                    config = self.config['ftune']
                    config['metric'] = 'loss'
                    self.jind_obj.ftune(train_data, config)
                    evaluate(self.jind_obj, train_data, name_tag="retrain", test_data_name=train_dataset_name)
                if self.config['align_target_to_source']:
                    train_data = self.train_data[self.train_data[BATCH] == self.source_dataset_name]
            
            # T-sne after training labeled batches
            # base_plot_name = "train{}-test{}".format(self.train_dataset_names, set(self.target_data[BATCH]))  
            # plot_name = "{}_adapt_retrain.pdf".format(base_plot_name)  
            # self.jind_obj.plot_tsne_of_batches(self.train_data.append(self.target_data), plot_name)  
            
            # Save Trained Model object
            print("\n[JindWrapper] Save Trained Model and val_stats object ")
            model_output_path = os.path.join(self.path, 'trained_models')

            # Ensure the directory exists
            os.makedirs(model_output_path, exist_ok=True)

            # Save each model separately
            for key, model_copy in self.jind_obj.model.items():
                torch.save(model_copy.state_dict(), os.path.join(model_output_path, f'{key}.pt'))

            # Save tha val stats calculated
            converted_val_stats = {key: val.tolist() for key, val in self.jind_obj.val_stats.items()}
            with open(os.path.join(model_output_path, 'val_stats_trained_model.json'), 'w') as f:
                json.dump(converted_val_stats, f)

            # Remove several `.pth` models  used as proxies for storing intermediate models during the training and tuning process
            remove_pth_files(self.path)
            
        else:
            print("\n[JindWrapper] Using an already trained classifier")
            train_data = self.train_data

        print("\n[JindWrapper] Training JIND for target dataset {}".format(set(target_data[BATCH])), "Train data shape = ", train_data.shape, "Target data shape = ", target_data.shape)
        predicted_label = perform_jind_training(self.jind_obj, train_data, self.target_data, self.config)
        print('predicted_label')
        print(predicted_label)

        if predicted_label is not None: 
            predicted_label.to_excel(os.path.join(self.path, "predicted_label_test_data.xlsx"))
        
        if model is None:
            timeline_name = "train{}-test{}".format(
                        [self.source_dataset_name] + [f'{len(self.intermediate_dataset_names)}_inter'],  # Lista concatenada con cadena formateada
                        set(self.target_data[BATCH])  # Conjunto (set)
                        )
            
            print("\n[JindWrapper] Plotting JIND model training timeline")
            raw_acc_per, eff_acc_per, mAP_per, rejected_per = plot_cmat_timeline(self.jind_obj.conf_matrix, self.path, timeline_name, num_datasets=len(self.intermediate_dataset_names)+2, cmat_print_counts=self.config['cmat_print_counts'])
            print("[JindWrapper] JIND training Done. Run Id = {}".format(self.path.split('/').pop()))
            return raw_acc_per, eff_acc_per, mAP_per, rejected_per
        else:
            print("[JindWrapper] JIND training Done. Run Id = {}".format(self.path.split('/').pop()))
            return None
        
    def get_dataset_with_least_batch_effect(self):
        if len(self.train_dataset_names) == 1:
            return self.train_dataset_names[0]

        dataset_to_rejection_map = {}
        print("[JindWrapper] Identifying dataset with least batch effect on {} from {}".format(set(self.target_data[BATCH]), self.train_dataset_names))

        for dataset in self.train_dataset_names:
            jind_obj = JindLib(self.train_data[self.train_data[BATCH] == dataset], self.path, self.config)
            jind_obj.train_classifier(config=self.config['train_classifier'])

            # jind_obj.evaluate(self.train_data[self.train_data[BATCH] == dataset])
            eval_result = jind_obj.evaluate(self.target_data, name="train{}_test{}_initial.pdf".format(dataset, set(self.target_data[BATCH])), return_log=True)

            rejection_count = len([f for f in eval_result["predictions"] if f == 'Unassigned'])
            dataset_to_rejection_map[dataset] = rejection_count
            print("[JindWrapper] Rejection count of dataset {} = {}".format(dataset, rejection_count))

        print(dataset_to_rejection_map)
        dataset_with_least_batch_effect = min(dataset_to_rejection_map, key=dataset_to_rejection_map.get)
        print("[JindWrapper] Rejection count of datasets = {}. Picking source dataset as {}".format(dataset_to_rejection_map, dataset_with_least_batch_effect))
        return dataset_with_least_batch_effect
