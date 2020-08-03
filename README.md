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

