# Running HMCNF for EC Number prediction

### Dependencies
rdkit
pytorch>=1.1.0
numpy
pandas

### Usage

`python test_on_molecules.py --smilesfile [testsmiles.txt] --outputfile [output.csv]`

smilesfile: text file containing the molecules (in smiles format) for which the user wishes to predict the EC Numbers. The file should contain one smiles string per line.

outputfile: name of the file (csv format) in which user wishes predictions to be stored.
The output csv file has:
	- rows: smiles (note that some input smiles may fail to convert to morgan2 fingerprints, so not all input smiles may be present in the outputfile)
	- columns: EC Numbers, 983 of them, i.e. all the ones that this model can make predictions for.
	- [smile, EC Number] = prediction for smile-ECNumber pair

For a working example, run:
`python test_on_molecules.py --smilesfile testsmiles.txt --outputfile output.csv`