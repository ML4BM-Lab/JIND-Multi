This script executes Jind Multi for Sergio Cámara to annotate a v4 dataset based on a smaller annotated v3 dataset.

The function to load the data is `load_scamara_data()`, which starts from the path where Cámara has the data:

### v3 smaller annotated dataset:
- It reads the v3 annotation data, converts it to a DataFrame, and plots the UMAPs.
- It selects the batches of v3 to be used for annotation.
- Variables 'batch' and 'labels' are assigned.
- Common genes and common labels between annotated batches.
- Heatmap of the labels per count and batch.

### v4 larger unannotated dataset (although we now have the ground truth):
- It reads the v4 annotation data and converts it to a DataFrame.
- It saves the ground truth in a file for result comparison. This file is saved in `./jind_multi/resources/data/scamara/results` and named `data_with_ground.csv`.
- Variable 'batch' is assigned, and 'labels' reflect the labels as 'Unassigned'.
- Common genes in unannotated batches.
- Get the intersection of common genes between v3 and v4 and filter.
- Reorder columns to move 'batch' and 'labels' to the end.
- Apply the lambda function to add "target_" before each name in the "batch" column (some batches' names repeat in v3 and v4).

In the `main()` function:
- Load the configuration of the `train_classifier`.
- Create the CUDA device.
- **Step 1:** Read the v3 and v4 data using `load_scamara_data()`.
- **Step 2:** Train JIND Multi for each Target batch:
    - From v4, select the target batch and concatenate this info with v3 (which is the training data).
    - Process the data together and filter the cells.
    - Split the data into train_data and test_data.
    - Initialize the JindWrapper object by selecting the source_dataset_name as one of the batch names in v3, and the output_path is `./jind_multi/resources/data/scamara/predictions/{target_batch}`.
    - Train Jind Wrapper. There are two options:
        - Option a. We don't have a model already trained in the results file `./jind_multi/resources/data/scamara/results`:
            ```python
            jind.train(target_data=test_data)
            ```
        - Option b. We have already trained Jind Multi with the training data and therefore have a model for each batch participating in the training. Here we use functions from the utility to load the models and save them in a dictionary so that they can fit the JindWrapper method, and we also load the val_stats, and subsequently apply Jind Multi only to train the target batch.
            ```python
            jind.train(target_data=test_data, model=model, val_stats=val_stats)
            ```
