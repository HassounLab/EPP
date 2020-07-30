# Multi-Label NN and HMCN-F data

### Molecules

We provide the molecules both as smiles and as maccs fingerprints:
    - smiles.csv contains the smiles
    - data.pkl contains the maccs fingerprints as a 2D numpy array, with the rows being parallel to the list of smiles,
      and the columns being the maccs features


### Molecule-Enzyme interaction labels

Files Pl1.pkl, Pl2.pkl, Pl3.pkl and Pl4.pkl contain the interaction labels for all molecules as 2D numpy arrays.

Each 2D-array's rows are parallel to the list of molecules, whereas the columns represent the Classes (Pl1), Subclasses (Pl2), Subsubclasses (Pl3) and EC Numbers (Pl4). Each column's index is mapped to a unique Class, Subclass, Subsubclass, or EC Number respectively via the python dictionaries class_index_dict.pkl, subclass_index_dict.pkl, subsubclass_index_dict.pkl, and ec_index_dict.pkl, found in the 'utils' folder.

For example, Pl4[5][6] is the interaction label for the 5th molecule and the 6th EC Number (both start from 0th).
Interaction labels are either 1.0 (positive) or 0.0 (unlabeled or inhibitor, both considered negative).


### Molecule-Enzyme interaction weights

Files PlN_sim_bal_weights.pkl, with N from 1 to 4, contain the similarity-adjusted label-balancing weights to be applied to each molecule-enzyme interaction label during training; these are the weights used to train the models in the paper.
Files PlN_sim_weights.pkl, with N from 1 to 4, contain the similarity weights only, without label-balancing.
FIles PlN_bal_weights.pkl, with N from 1 to 4, contain only label-balancing weights, without similarity-based adjustment.

The format of the weights is the same as the labels. Please refer to the paper, section 2.3, for a full explanation of how these weights were computed.

### Inhibitor interactions
Files PlN_sim_bal_weights_inh.pkl, with N from 1 to 4, contain the similarity-adjusted label-balancing weights, with the difference that there is no similarity-based adjustment for inhibitor molecules; these are the weights used to train the models in the paper.
Files PlN_sim_weights_inh.pkl, with N from 1 to 4, contain the similarity weights only, with the weights of inhibitor molecules set to 1.0.
Please refer to the paper, section 2.3, for a full explanation of how these weights were computed.

##### To find out which Molecule-Enzyme interactions are inhibitor interactions

If a label is 0.0 in PlN.pkl, and the corresponding weight in PlN_sim_weights_inh.pkl is 1.0, then the interaction is an inhibitor interaction.

### Global labels and weights

All Pg_\*.pkl files contain their respective Pl1_\*.pkl, Pl2_\*.pkl, Pl3_\*.pkl, Pl4_\*.pkl, stacked horizontally in this order.

### To map column indices to the corresponding enzymes

Use the python dictionaries saved as class_index_dict.pkl, subclass_index_dict.pkl, subsubclass_index_dict.pkl, ec_index_dict.pkl, contained in the 'utils' folder.
    - EnzymeLevel_index_dict[enzyme] = index
    - EnzymeLevel_index_dict[index] = enzyme


### Training and Testing datasets

indexes_for_splitting.pkl contains an array of randomized indices to be applied to data.pkl, and to the labels and weights used. Then, the first 20% of the rows are saved for testing, whereas all the other ones are used for training. This generates the Full Test Set and the Training Set.

The Inhibitor Test Set and the Unlabeled Test Set are generated on-the-fly. Please refer to the paper, section 3, for details on how to generate these sets.
