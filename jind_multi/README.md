# JIND-Multi
JIND-Multi is an extension of the JIND framework designed for the automated annotation of single-cell RNA sequencing (scRNA-Seq) data and scATAC-Seq data. JIND-Multi facilitates the transfer of cell-type labels from multiple annotated datasets and provides the option to mark cells as "unassigned" if the model's predictions are not reliable. By leveraging a large number of annotated datasets, JIND-Multi enhances the accuracy of annotations for unlabeled datasets while minimizing rejection rates (unassigned cells).

## Package Structure
The `JIND-Multi` package is organized as follows:

### `jind_multi/`  # Package Directory

- **`__init__.py`**  
  Initializes the package, allowing `jind_multi` to be recognized as a module in Python.

- **`core.py`**  
  Contains the primary function for training JIND-Multi. Implements the following stages:
  1. Data loading and normalization.
  2. Splitting the data into training and testing sets.
  3. Creating the JIND-Multi object.
  4. Training the JIND-Multi model.

- **`config_loader.py`**  
  This module manages the configuration for training JIND-Multi. It includes parameters such as:
  - Number of features to consider.
  - Minimum cell type population.
  - Maximum cells per dataset.
  - Paths to different datasets.
  - Training settings and CUDA utilization.

- **`utils.py`**  
  Provides utility functions for:
  - Saving and loading models.
  - Generating plots and performing data preprocessing.
  - Additional auxiliary functions to facilitate data and model handling.

- **`jind_wrapper.py`**  
  Defines the `JindWrapper` class, the core component of the package. This class:
  - Manages the JIND-Multi object and utilizes JindLib to train the classifier and encoder using the source dataset.
  - Adapts various labeled datasets to the latent space of the source dataset.
  - Calls JindLib to train the GAN and infer labels for the target dataset.
  - Generates final results including tables and plots, and saves the trained models.

- **`data_loader.py`**  
  Responsible for:
  1. Reading an ANN object and adding batch and labels columns.
  2. Selecting common genes and labels between batches.
  3. Processing data for training and testing purposes.
  - Includes a custom data loader class tailored for specific training needs.