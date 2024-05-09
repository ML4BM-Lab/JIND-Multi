
ConfigLoader.py -> contains the configuration for training the classifier, the GAN, and the fine-tuning, as well as the paths to the different datasets loaded in DataLoader.py and the number of genes (num_features) and minimum number of cells per batch set used.

DataLoader.py -> contains the code necessary to load the different datasets, as well as to preprocess them (ensuring that the data is normalized and log-transformed, applying dimensionality reduction with config['num_features'], and filtering cells with min_cell_type_population=config['min_cell_type_population'].

JindLib.py -> contains the JindLib class, which primarily trains the classifier, the fine-tuning, and the training on the target.

JindWrapper.py ->

If the user does not provide a source, it applies
