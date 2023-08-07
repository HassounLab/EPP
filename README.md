# Enzyme Promiscuity Prediction using hierarchy-informed multi-label classification


-------------------------------------------------------------------------

Requirements:
  - numpy
  - pandas
  - matplotlib
  - sklearn
  - rdkit
  - pytorch (v. 1.4.0 was used)

-------------------------------------------------------------------------

## HMCN-F

### Data

Found in **data/HMCNF_data**.

### Training and Testing

run `python HMCNF.py --similarity [True] --inhibitors [False] --output_folder [./]`

This will run a Random Search cross validation over 12 sets of hyperparameters, then re-train the model with the hyperparameters' selection that yields the best Mean AP score during cross validation.
The model, together with a plot showing AUROC, AP and R-PREC across epochs on the test set, are saved in the user-provided output_folder.

Arguments:
  - *similarity*: bool; whether to use similarity-based information in weighting the training enzyme-molecule pairs. Automatically true if *inhibitors* is set to True.
  - *inhibitors*: bool; whether to use inhibitor information in weighting the training enzyme-molecule pairs. This automatically enables similarity-based weighting as well.
  - *output_folder*: str; location where to store trained model and results plot.
  **TODO: add *ratio* as an argument.**

## MultiLabel-NN

### Data

Found in **data/HMCNF_data**.

### Training and Testing

run `python MultiLabelNN.py --similarity [True] --inhibitors [False] --output_folder [./]`

All other specifications are the same as HMCNF.py.

## Hierarchical RF and No-Share RF

Hierarchical RF and No-Share RF are part of the same class and are trained together

### Data

See the README in **data/tree_data** on how to convert the data in **data/HMCNF_data** to the right format for Hierarchical RF and No-Share RF.

### Training

run `train_tree_and_flat.py --similarity [True] --inhibitors [False] --ratio [all] --output_folder [./]`

This will run the training of Hierarchical RF (the 'Tree' model) and of No-Share RF (the 'Flat' model).

All arguments are the same as for HMCNF.py.

### Testing

run `test_tree_and_flat.py --inhibitors [False] --similarity [True] --ratio [all] --test_set [Full] --estimator [Tree] --model_folder [./] --output_template [results]`

Arguments:
  - *inhibitors*: see above. Needed in selecting the right model to test.
  - *similarity*: see above. Needed in seleting the right model to test.
  - *ratio*: see above. Needed in selecting the right model to test.
  - *test_set*: str; string that identifies the kind of test set to use. See paper for full details. Default is the Full Test Set. **TODO** add extraction of the inhibitors test set to this repo.
  - *estimator*: str; string that identifies whether user wishes to test Hierarchical RF ("Tree") or No-Share RF ("Flat")
  - *model_folder*: str; path to folder containing the model of interest
  - *output_template*: str; path + template (initial part of filename) where to store the results.


## On Realistic data split

To train and evaluate the models under the realistic data split, the following two steps need to be taken:

### Splitting the molecules

We provide the indices for the "realistic" split of the data, as defined in our paper via UPGMA clustering of the molecules, and choosing the test molecules as the outliers. The files are the following:
- Training + Validation: `data/HMCNF_data/train_i_UPGMAcluster800.pkl`. Slices the full dataset.
- Training only: `data/HMCNF_data/train_for_val_i_UPGMAcluster800.pkl`. Slices the training + validation dataset (i.e. apply it after `train_i_UPGMAcluster800.pkl`).
- Validation only: `data/HMCNF_data/validation_i_UPGMAcluster800.pkl`. Slices the training + validation dataset (i.e. apply it after `train_i_UPGMAcluster800.pkl`).
- Test: `data/HMCNF_data/test_i_UPGMAcluster800.pkl`. Slices the full dataset.

### Subsetting the labels (aka Reduced set)

As described in the paper, the realistic split divides the molecules in such a way that several EC numbers do not have enough molecules associated with them for training and/or testing, so models could be trained only for 680 EC Numbers. Furthermore, when splitting the training data further into training and validation for the purposes of hyperparameter optimization, 680 is further reduced to 561.

We provide a script to subset the Full dataset into Reduced and Reduced-for-validation datasets. The following commands will generate the appropriate data files:
```bash
cd data/HMCNF_data
python make_realistic_split_data.py --similarity [True] --inhibitors [False]
```


## On Inhibitors

To identify all the inhibitors in our dataset, we provide an `(N x E)` boolean matrix - where `N` is the total number of molecules and `E` is the total number of ECs - in `data/HMCNF_data/PL4_inh.pkl`. This matrix identifies the inhibitor molecules of each EC number, and can be easily used to subset the Full Test Set set to create the Inhibitor Test Set.
