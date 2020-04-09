'''
    The user must give a valid dictionary that maps molecules in whatever representation 
    they want to rdkit Fingerprint objects.
    The kind of fingerprint must match between training data and testing data. Our training
    data uses maccs fingerprints.
'''

from rdkit import DataStructs
from rdkit.Chem import MACCSkeys

from utils import pickle_load

class SimilarityModel(object): 
    def __init__(self, training_data_filepath, to_maccs_table_filepath):
        '''
        training_data_filepath must be a .pkl file.
        '''
        self.training_data_filepath = training_data_filepath
        self.training_data = pickle_load(training_data_filepath)
        self.to_maccs_table = pickle_load(to_maccs_table_filepath)
    
    def test(self, test_data_filepath, EC=None):
        '''
        test_data_filepath must be a .pkl file.
        If EC is None, test() expects test_data_filepath to point to a dictionary where keys are EC Numbers
        and values are lists of molecules.
        Otherwise, if enzyme is an EC Number, test() expects test_data_filepath to point to a list of molecules.
        Some molecules might not be pr
        '''
        test_data = pickle_load(test_data_filepath)
        if EC is None: # dict case
            assert type(test_data) == dict
        
            results = {}
            for enzyme in test_data:
                if enzyme not in self.training_data:
                    continue

                results[enzyme] = {}
                # make bitstrings for train and test data

                train_pos = self.training_data[enzyme]["molecules"]
                test = test_data[enzyme]["molecules"]

                # compute similarities - need to convert each molecule to its fp representation
                mean_scores = []
                max_scores = []
                mols = []
                for test_mol in test:
                    test_fp = self.to_maccs_table[test_mol]
                    if test_fp is None:
                        continue
                    temp_scores = []
                    for pos_mol in train_pos:
                        pos_fp = self.to_maccs_table[pos_mol]
                        if pos_fp is None:
                            continue
                        temp_scores.append(DataStructs.FingerprintSimilarity(pos_fp, test_fp))

                    if len(temp_scores) == 0:
                        continue
                    mean_scores.append(np.mean(temp_scores))
                    max_scores.append(np.max(temp_scores))
                    mols.append(test_mol)
                
                results[enzyme]["Mean"] = mean_scores
                results[enzyme]["Max"] = max_scores
                results[enzyme]["Molecules"] = mols
                
        else: # individual enzyme case
            assert type(enzyme) == str
            if enzyme not in self.training_data:
                raise Exception(enzyme + " is not in training data.")
            
            results = {enzyme: {}}
            # make bitstrings for train and test data
            train_pos = self.training_data[enzyme]["molecules"]
            test = test_data

            # compute similarities - need to convert each molecule to its fp representation
            mean_scores = []
            max_scores = []
            mols = []
            for test_mol in test:
                test_fp = self.to_maccs_table[test_mol]
                if test_fp is None:
                    continue
                temp_scores = []
                for pos_mol in train_pos:
                    pos_fp = self.to_maccs_table[pos_mol]
                    if pos_fp is None:
                        continue
                    temp_scores.append(DataStructs.FingerprintSimilarity(pos_fp, test_fp))

                if len(temp_scores) == 0:
                    continue
                mean_scores.append(np.mean(temp_scores))
                max_scores.append(np.max(temp_scores))
                mols.append(test_mol)
            
            results[enzyme]["Mean"] = mean_scores
            results[enzyme]["Max"] = max_scores  
            results[enzyme]["Molecules"] = mols         
            
        return results
        