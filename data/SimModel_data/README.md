# Data for models Similarity Mean and Similarity Max

**pos_dict_train_SimModel.pkl**
    - Our data used for training, contained in a Python dictionary where keys are EC Numbers and values are lists of molecules, where each molecule is represented by a bitstring representation of the molecule's maccs fingerprint

**bitstrings_maccs_table.pkl**
    - Python dictionary mapping maccs bitstrings to rdkit maccs fingerprint objects. For evaluation using this model to be successful, this dictionary must all molecules in training and testing. The current verion contains all molecules in pos_dict_train_SimModel.pkl, pos_dict_test.pkl, unl_dict_test.pkl
