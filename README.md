# JIND-Multi  
<!-- #### Publication:   -->
We introduce JIND-Multi, an extension of the JIND framework for the automated annotation of single-cell RNA sequencing (scRNA-Seq) data ([Goyal et al., 2022](https://academic.oup.com/bioinformatics/article/38/9/2488/6543609)), that allows transferring cell-type labels from several annotated datasets. Notably, JIND-Multi is also applicable for the annotation of scATAC-Seq data. Moreover, similarly to its predecessor, JIND-Multi has the option to mark cells as "unassigned" if the model does not produce reliable predictions, i.e., below some pre-computed cell type specific thresholds.

The proposed approach JIND-Multi can leverage a large number of annotated datasets, e.g., those that compose an atlas, making the annotation of unlabeled datasets more precise and with lower rejection rates (unassigned cells). We provide an efficient implementation of JIND-Multi that is publicly available and ready to use by the community.

<p align="center">
    <img src="JIND.pdf" width="700" alt="PDF Image">
</p>

## Prerequisites
- Linux or macOS
- Miniconda
- Python 3.6 or higher (tested on 3.6.8 and 3.7.11)
- CPU or NVIDIA GPU + CUDA CuDNN

## Installation
<!-- ```bash
git clone https://github.com/ML4BM-Lab/JIND-Multi.git
cd JIND-Multi
conda env create -f environment.yml
conda activate jind
``` ESTO HAY QUE VERLO BIEN --> 

## Data
The datasets can be downloaded from the following links:  https://doi.org/10.5281/zenodo.11098805

# Executing JIND-Multi
You have two options to execute the JIND-Multi framework: directly running the Python script or submitting it as a job in a cluster.

### Option 1: Run the Python Script Directly
For example, if we want to execute the `Brain Neurips` dataset, we would do it as follows:

```bash
cd main
conda activate jind
python -u Main.py \
  --PATH /path/to/your/JIND-Multi_folder/../resources/All_human_brain.h5ad \
  --BATCH_COL "batch" \
  --LABELS_COL "label" \
  --SOURCE_DATASET_NAME "C4" \
  --TARGET_DATASET_NAME "C7" \
  --OUTPUT_PATH /path/to/save/results \
  --TRAIN_DATASETS_NAMES "['AD2', 'ADx1', 'ADx2', 'ADx4']" \
  --NUM_FEATURES 5000 \
  --MIN_CELL_TYPE_POPULATION 100 \
  --USE_CUDA True
```

where,
- **`PATH`**: (string) path where you store the `.h5ad` file with your data.
- **`BATCH_COL`**: (string) name of the column with the information of the different batches or donors in your AnnData object.
- **`LABELS_COL`**: (string) name of the column with the different cell types in your AnnData object.
- **`SOURCE_DATASET_NAME`**: (string) Optional. name of the source batch. Alternatively, if no batch is specified, JIND-Multi will select as source the batch that produces the least amount of rejected cells on the target batch when used as source in JIND (i.e., without additional intermediate batches).
- **`TARGET_DATASET_NAME`**: (string) name of the target batch.
- **`OUTPUT_PATH`**: (string) path where you want to save the model performance results. 
<!-- AND trained model ???. -->
- **`TRAIN_DATASETS_NAMES`**: (string) Optional. This setting allows you to specify the order of intermediate datasets used for training. You do not need to include the source dataset name here, as it will be selected automatically. If you leave this option empty, the system will use the intermediate datasets in the order they appear in the data.
- **`NUM_FEATURES`**: (int) Optional. Number of genes to consider for modeling, default is 5000.
- **`MIN_CELL_TYPE_POPULATION`**: (int) Optional. For each batch, the minimum number of cells per cell type necessary for modeling. If this requirement is not met in any batch, the samples belonging to this cell type are removed from all batches, default is 100.
- **`USE_CUDA`**: (bool) Optional. Whether to use CUDA if available, default is True.

### Option 2: Submit as a Job in a Cluster
If you prefer, you can submit the job using the provided `main.sh` script from the cluster directory. Make sure to specify the correct arguments for your dataset in the script before submitting the job.

```bash
cd cluster
sbatch main.sh
```

En `OUTPUT_PATH` se guardan los siguientes outputs:
- predicted_label_test_data.xlsx : tabla con las predicciones sobre el target data, indicando para cada una de las muestras la probabilidad calculada por el modelo de que pertenezca a cada uno de los celltypes. Se indica en la columna `raw_predictions`, la celltype con mayor probabilidad antes de aplicar el threshold particula para cada celltype y `predictions` el celltype prediction después de aplicar el filtrado.
- Los modelos particulares finales entrenados por cada batch anotado en formato .pt y un target.pth con el modelo entrenado para el target batch. También encontrarás varios modelos .pth guardados pero estos son ficheros intermedios usados por el método como proxy para guardar modelos intermedios durante el proceso de entrenamiento y tuneado del classificador y el encoder. No hacer uso de ellos. Además, en val_statds_trained_model.json se guardan las predicciones sobre el test de validacion a la hora de entrenar el classifier que se utilizan poder computar los thresholds.
- los resultados de la peformance del modelo sobre el source batch, los datasets intermedios y el validation set después de entrenar el clasificador y hacer los distintos ftuning mediante matrices de confusion en el que se indica el número de muestras, cuantas han sido rejected y cuantas correctamente predichas con los porcentajes del accuracy antes (raw) y después de aplicar el threshold (eff), cuantas incorrectas y la mean average precision (mAP) por celltype.

Para el source batch se muestran las matrices de confusión después de entrenar el classifier, después del tuning y después del ultimo tuning (retrain). Para el resto de batches intermedios se indica los resultados antes de alinear las muestras al espacio latent del source ("initial"), después del alineamiento ("adapt") y tras el ultimo tuning del classifier y el encoder ("retrain"). En caso de que el target batch disponga de labels también se sacarían las matrices de confusión antes de entrenar el generative Adversarial network (GAN), después de entrenar la GAN y después del tuning del encoder y el clasificador usando las muestras con mayor confianza. El historial de estos matrices de confusión también se guarda en un fichero pdf con esta forma: train['nombre_source_batch', 'number_inter_batches]-test{'nombre_target_batch}.pdf.

- Por defecto, el usuario también dispondrá de tsne plots con la proyección coloreado por batch o labels del espacio latente generado al pasar las muestras por el encoder en 4 distintos momentos:

1) tras entrenar el encoder y el classifier con el source batch ("after_train_classifier").
2) después de eliminar el batch effect de cada uno de los batches intermedios y antes de inferir sobre el target batch ("initial").
3) después de alinear el target batch al latent code del source con el entrenamiento de la GAN.
4) después de hacer el tuning del encoder y el classifier para el target batch.

Re-using an already trained modeL: se puede reusar los modelos ya entrenados simplemente indicando en un nuevo run como indicado en el Option 1 y Option 2 el mismo `OUTPUT_PATH` donde están guardados los modelos, JIND-Multi los detectará y cargará los modelos con los pesos ya entrenados para inferir sobre el nuevo target batch. 

In the `OUTPUT_PATH`, the following outputs are saved:

- A table with the predictions on the target data (**predicted_label_test_data.xlsx**:), indicating for each sample the probability calculated by the model for each cell type. The `raw_predictions` column shows the cell type with the highest probability before applying the cell type-specific threshold, and the predictions column shows the predicted cell type after filtering.
The final trained models for each annotated batch in `.pt` format, and a `target.pth` file with the trained model for the target batch. Additionally, several `.pth` models are saved, which are used as proxies for storing intermediate models during the training and tuning process. Do not use these intermediate files. The file `val_stats_trained_model.json` contains predictions on the validation test set used to compute the thresholds.

- The model performance results on the source batch, intermediate datasets, and validation set after training the classifier and various fine-tuning steps. These results include confusion matrices indicating the number of samples, how many were rejected, and how many were correctly predicted with the accuracy percentages before (raw) and after applying the threshold (eff), as well as incorrect predictions and the mean average precision (mAP) per cell type.
For the source batch, confusion matrices are shown after training the classifier, after tuning, and after the final tuning (retrain). For the intermediate batches, results are shown before aligning the samples to the latent space of the source ("initial"), after alignment ("adapt"), and after the final tuning of the classifier and encoder ("retrain"). If the target batch has labels, confusion matrices are also provided before training the generative adversarial network (GAN), after training the GAN, and after tuning the encoder and classifier using the most confident samples. The history of these confusion matrices is also saved in a PDF file named train[source_batch_name, number_inter_batches]-test[target_batch_name].pdf.

- By default, the user will also have t-SNE plots with projections colored by batch or labels of the latent space generated by passing the samples through the encoder at four different moments:

1) After training the encoder and classifier with the source batch ("after_train_classifier").
2) After removing the batch effect from each intermediate batch and before inferring on the target batch ("initial").
3) After aligning the target batch to the latent code of the source with the GAN training.
4) After tuning the encoder and classifier for the target batch.

## Re-using an Already Trained Model
You can reuse already trained models by specifying the same `OUTPUT_PATH` in a new run as indicated in `Option 1` and `Option 2`. JIND-Multi will detect and load the models with the pre-trained weights to infer on the new target batch.

## Example
You can find an example of executing JIND-Multi, explaining in more detail the data processing and the internal functioning of the method, in the `./example` folder.

# Compare JIND Methods
To compare the annotation performance of JIND, JIND-Multi, and JIND-Combined (merging all annotated datasets into one, without correcting for batch effect) for any target dataset with known true labels, you can execute the `compare_methods.sh` script.

# Additional Information
In the `./main` folder, you will find another README explaining in more detail each of the Python scripts of the JIND-Multi method.


AQUIII!! TRADUCE TODO BIEN Y PONLO LIMPITO! SUBELO A GITHUB Y LUEGO CLONA EL REPOSITORIO EN LOCAL PARA HACER EL ENVIRONMENT BIEN Y EL JUPYTER NOTEBOOK
<!-- 
## Applying the Model
Within the `main` folder, you will find a file named `main.py`, which carries out all the necessary steps to annotate a dataset using various differently annotated batches. To execute it, simply run the following Python script.

``` shell
python Main.py -dt pancreas -s 0 -t 2 -p /path_to/jind_multi
```
Here, `-dt` represents the name of the dataset to be loaded, `-s` signifies the primary source batch among the available annotated batches, `-t` denotes the target batch to annotate, and `-p` points to the path of the 'jind_multi' folder. 

Let's delve into a detailed explanation of what this code does:

First, the data is loaded and normalized using the `load_data` function from the `DataLoader.py` script. It loads the `ann` object from the path set in `ConfigLoader.py`, reads it as a dataframe with dimensions `num_cells x n_genes`, assigns the `batch` and `labels` columns, and performs common gene and label operations across different batches. Subsequently, it normalizes and transforms the data, performs dimension reduction on the top 5000 genes with the highest variance, and filters out cells using `filter_cells`, a function from `Utils.py` that removes cell types that are not sufficiently represented in any of the annotated batches. Additionaly, a `min_cell_type_population` is defined with a default value as 100, but it can be modified. Data processing is conducted with both annotated data and the target batch together.

```python
data = load_data(data_type=args.DATA_TYPE)
```
Next, the data is divided into `train_data`, containing both the source batch (primary annotated batch) and the remaining intermediate batches, and `test_data`, representing the target batch to annotate.

```python
train_data = data[data['batch'] != args.TARGET_DATASET_NAME]
test_data = data[data['batch'] == args.TARGET_DATASET_NAME]
```

Afterwards, the JIND Multi object is created, and we fit the `train_data`, the name of the source batch (`source_dataset_name`), and the path where we want to save the results (`output_path`).

```python
jind = JindWrapper(train_data=train_data, source_dataset_name=args.SOURCE_DATASET_NAME, output_path=args.PATH_WD+'/output/'+ args.DATA_TYPE)
```

Finally, JIND Multi is trained. The encoder and classifier are trained with the source batch, minimizing the categorical cross-entropy loss. Then, for each intermediate batch, `perform_domain_adaptation` function is executed to adapt each intermediate dataset to the latent space of the source. This involves creating a custom model for each data set, inserting two NN blocks with an arquitecture similar to JIND+'s generator. This training involves using the labels and minimizing the categorical cross-entropy loss while keeping the parameters of the encoder and classifier fixed.

After each adaptation, the encoder and classifier undergo a `ftune` using the batches trained up to that point. Once all batches are adapted, a final tuning is performed. 

The particular models for each training set are saved so that they can be reused for different target batches without needing to re-run Jind Multi on the same train data. Additionally, `val_stats` contains predictions and labels used to calculate specific thresholds for each cell type during training are saved, which application ensures that we don't generate predictions with low confidence.

Finally, the trained encoder and classifier are applied to the target batch. This involves aligning the latent space of the target using a GAN (since this time we do not have labels). The main advantage of JIND Multi is that this method allows to use more real samples to train the GAN, as we have different annotated batches adapted to the same latent space. Sequentially, the Discriminator is fed with target samples along with samples from one of the annotated batches each time, and the Generator is improved through competitive training.

```python
jind.train(target_data=test_data)
```

As mentioned earlier, if you have already trained JIND Multi on a `train_data` to annotate one target batch, you can reload these models and use them to annotate another batch. `file_paths` contains a list of paths to the custom models trained for your train_data.

```python
# Load the trained models
model = load_trained_models(file_paths, train_data, source_dataset_name, device)
# Load the val_stats
val_stats = load_val_stats(model_input_path, 'val_stats_trained_model.json') 
# Do Jind
jind.train(target_data=test_data, model=model, val_stats=val_stats)
```





