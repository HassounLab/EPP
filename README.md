# Enzyme Promiscuity Prediction using hierarchy-informed multi-label classification


-------------------------------------------------------------------------

Requirements:
  - numpy
  - sklearn
  - rdkit
  - pytorch (v. 1.4.0 was used)

-------------------------------------------------------------------------

## HMCN-F

### Data

Found in **data/HMCNF_data**

### Training

run `python HMCNF.py --similarity [True] --inhibitors [False] --output_folder [./]`

This will run a Random Search cross validation over 12 sets of hyperparameters, then re-train the model with the hyperparameters' selection that yields the best Mean AP score during cross validation.
The model, together with a plot showing AUROC, AP and R-PREC across epochs on the test set, are saved in the user-provided output_folder

Arguments:
  - *similarity*: bool; whether to use similarity-based information in weighting the training enzyme-molecule pairs. Automatically true if *inhibitors* is set to True.
  - *inhibitors*: bool; whether to use inhibitor information in weighting the training enzyme-molecule pairs. This automatically enables similarity-based weighting as well.

## MultiLabel-NN

### Data

Found in **data/HMCNF_data**

### Training

run `python MultiLabelNN.py --similarity [True] --inhibitors [False] --output_folder [./]`

All other specifications are the same as HMCNF.py.

