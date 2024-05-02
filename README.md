# JIND-Multi
##  
#### Publication:  

<!-- ABOUT THE PROJECT -->
## Description
<p align="center">
    <img src="jind_multi.png" width="700" alt="PDF Image">
</p>

We introduce JIND-Multi, an extended version of the JIND framework designed for label transfer across datasets using multiple labeled datasets. When applied to single-cell RNA sequencing (scRNA-Seq) data, 
JIND-Multi demonstrates a significant reduction in the proportion of unclassified cells while preserving accuracy and performance equivalent to JIND. Notably, our model achieves robust and accurate results 
in its inaugural application to scATAC-Seq data, showcasing its effectiveness in this context.

#### Publication: 

### Built With
*   <a href="https://www.python.org/">
      <img src="https://www.python.org/static/community_logos/python-logo.png" width="110" alt="python" >
    </a>
*   <a href="https://pytorch.org/">
      <img src="https://pytorch.org/assets/images/pytorch-logo.png" width="105" alt="pytorch" >
    </a>

## Clone repository & create the Conda environment
To crate a Conda environment using the provided environment.yml, follow these steps:

```bash
git clone https://github.com/ML4BM-Lab/JIND-Multi.git
cd jind
conda env create -f environment.yml
conda activate jind
```
## Data
The datasets can be downloaded from the following links:  https://doi.org/10.5281/zenodo.11098805

## Applying the Model ('\main')
Within the 'main' folder, you will find a file named main.py, which carries out all the necessary steps to annotate a dataset using various differently annotated batches. To execute it, simply run the Python script.

shell
Copy code
python Main.py -dt pancreas -s 0 -t 2 -p /home/jsanchoz/data/josebas/JIND_Iterative/JIND-continual_integration/jind_multi
Here, -dt represents the name of the dataset to be loaded, -s signifies the primary source batch among the available annotated batches, -t denotes the target batch to annotate, and -p points to the path of the 'jind_multi' folder. These arguments are parsed when calling the Main.py script as follows:

python
Copy code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JindMulti')
    parser.add_argument('-dt', '--DATA_TYPE', type=str, required=True, help='Dataset name') 
    parser.add_argument('-s', '--SOURCE_DATASET_NAME', type=str, help='Name or ID of source dataset') 
    parser.add_argument('-t', '--TARGET_DATASET_NAME', type=str, required=True, help='Name or ID of target dataset') 
    parser.add_argument('-p', '--PATH_WD', type=str, required=True, help='Path to jind_multi folder') 
    args = parser.parse_args()
    main(args)
Let's delve into a detailed explanation of what this code does:

python
Copy code
# 1) load data and normalize
data = load_data(data_type=args.DATA_TYPE)
This line loads and processes the desired dataset using the load_data function from the DataLoader.py script. It loads the 'ann' object from a path set in the ConfigLoader.py, reads it as a dataframe with dimensions num_cells x n_genes, assigns the 'batch' and 'labels' columns, and performs common gene and label operations across different batches. Subsequently, it normalizes and transforms the data, performs dimension reduction on the top 5000 genes with the highest variance, and filters out cells using 'filter_cells', a function from Utils.py that removes cell types that are not sufficiently represented in any of the annotated batches. The 'min_cell_type_population' is defined as 100 but can be modified. Data processing is conducted with both annotated data and the target batch together.

Next, the data is divided into train_data, containing both the source batch (primary annotated batch) and the remaining intermediate batches, and test_data, representing the target batch to annotate.

python
Copy code
# 2) Divide in train and test
train_data = data[data['batch'] != args.TARGET_DATASET_NAME]
test_data = data[data['batch'] == args.TARGET_DATASET_NAME]
Afterwards, the JIND Multi object is created, and we input the train_data, the name of the source batch, and the path where we want to save the results.

python
Copy code
# 3) Create the Jind Multi object
jind = JindWrapper(train_data=train_data, source_dataset_name=args.SOURCE_DATASET_NAME, output_path=args.PATH_WD+'/output/'+ args.DATA_TYPE)
Finally, JIND Multi is trained for the target batch. Firstly, the encoder and classifier are trained with the source batch, minimizing the categorical cross-entropy loss. Then, for each intermediate batch, 'perform_domain_adaptation' is executed to adapt each intermediate dataset to the latent space of the source. This involves creating a custom model for each batch, inserting two NN blocks with the same architecture as the Generator used in JIND+, training this model using the labels and minimizing the categorical cross-entropy loss while keeping the parameters of the encoder and classifier fixed.

After each adaptation, the encoder and classifier undergo 'ftune' using the batches trained up to that point. Once all batches are adapted, a final 'ftuning' is performed. The particular models for each training batch are saved so that they can be reused for different target batches at various times without needing to rerun Jind Multi on the train data. Additionally, 'val_stats' containing predictions and labels used to calculate specific thresholds for each cell type are saved to ensure predictions with low confidence are not provided.

Finally, the trained encoder and classifier are applied to the target batch. This involves aligning the latent space of the target using a GAN since this time we do not have labels, and with JIND Multi, we have more real samples to train from as we have different annotated batches adapted to the same latent space. Sequentially, the Discriminator is fed with target samples along with samples from one of the annotated batches each time, and the Generator is improved through competitive training.

python
Copy code
# 4) Train Jind Multi
jind.train(target_data=test_data)
As mentioned earlier, if you have already trained JIND Multi once on a target batch, you can reload these models and use them to annotate another batch. 'file_paths' contains a list of paths to the custom models trained for your train_data.

python
Copy code
# 4) Train Jind Multi
# Load the trained models
model = load_trained_models(file_paths, train_data, source_dataset_name, device)
# Load the val_stats
val_stats = load_val_stats(model_input_path, 'val_stats_trained_model.json') 
# Do Jind
jind.train(target_data=test_data, model=model, val_stats=val_stats)

        
   
       






