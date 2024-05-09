def get_default_config():
    config = {
        "ftune_intermediate": True, # default was True
        "retrain_intermediate": True, # default was True
        'align_target_to_source': False, 
        "plot_tsne": False,
        "cmat_print_counts": True,
        'save_results_to_sheets': True,
        "train_classifier": {
            "val_frac": 0.2,
            "seed": 0,
            "batch_size": 128,
            "cuda": True,
            "epochs": 15
        },
        "GAN": {
            "seed": 0,
            "batch_size": 128,
            "cuda": True,
            "epochs": 15,
            "epochs_da": 15,
            "gdecay": 1e-2,
            "ddecay": 1e-3,
            "maxcount": 7,
            "val_frac": 0.1,
            "version": "domain_adapt"
        },
        "ftune": {
            "version": "ftune",
            "val_frac": 0.1,
            "seed": 0,
            "batch_size": 32,
            "cuda": True,
            "epochs": 15,
            "epochs_intermediate": 5,
            "retrain_encoder": True,
            "use_all_labels": True,
            "mini_batch_len": 'MAX',
            'mini_batch_splits': 1,
            'ftune_retrain_model_epochs': 5,
            'metric': 'accuracy'    # Possible values: accuracy, loss
        },
        "data": {
            "count_normalize": True,
            "log_transformation": True,
            "num_features": 5000,
            "min_cell_type_population": 18, # by default put at least 100 always. 
            # 20 for "NSCLC_lung", 5 for "pancreas" when using source 3, 100 for "human_brain", 18 for "bmmc_ATAC", 100 for all ATAC2 tissues (liver atac2 35)

            "max_cells_for_dataset": 50000,
            "test_data_path": "../resources/data/test/",
            "human_brain_neurips_data_path": "../resources/data/human_brain/All_human_brain.h5ad",
            "nsclc_lung_data_path": "../resources/data/NSCLC_lung/NSCLC_lung_NORMALIZED_FILTERED.h5ad",
            "pancreas_data_path": "../resources/data/pancreas/pancreas.h5ad",
            "brca_data_path": "../resources/data/breast/BRCA_data.h5ad",
            "bmmc_atac_path": "../resources/data/bmmc/data_multiome_annotated_BMMC_ATAC.h5ad",
            "bmmc_gex_path": "../resources/data/bmmc/data_multiome_annotated_BMMC_GEX.h5ad",
            "atac2_data_path": "../resources/data/atac_2/Filtered_Norm_Scaled_data_Annotated_subseted.h5ad",
            "tissue_data_path": "../resources/data/atac_2/tissue_sample_norm_scaled_data_annotated.h5ad",
            "brain_data_source": "MouseV1",
            "brain_data_source_gene_nums": 8000
        }
    }
    return config


def get_config(user_conf={}):
    config = get_default_config()
    for key in user_conf.keys():
        if key in config.keys():
            config[key] = {**config[key], **user_conf[key]} if (type(config[key]) is dict) else user_conf[key]
        else:
            config[key] = user_conf[key]
    return config