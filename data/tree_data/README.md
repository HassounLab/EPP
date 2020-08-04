## To get data for Tree and Flat classifiers

Data for the Tree and Flat classifiers is adapted from the data formatted for the HMCNF.

First, convert label and weight matrices into dictionaries for positve and for negative data.

`python convert_from_NN_to_tree_data.py --inhibitors False`


Then, adapt the dictionaries to be used by the Tree and Flat classifiers.

`python create_dataset.py --inhibitors False --ratio all`

This will create the training file and testing file with a specific P/T value to be fed to the 
Tree and Flat classifiers.
