'''
    The user must give a valid dictionary that maps molecules in whatever representation 
    they want to rdkit Fingerprint objects.
    The kind of fingerprint must match between training data and testing data. Our training
    data uses maccs fingerprints, represented with their bitstrings.
'''

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys

from utils import pickle_load

def arrays_to_bitstrings(arrays):
    bitstrings = []
    for array in arrays:
        bitstrings.append(''.join([str(int(x)) for x in array]))
    return np.array(bitstrings)

class SimilarityModel(object): 
    def __init__(self, training_data_filepath, to_fpobj_table_filepath, training_mols, test_mols):
        '''
        training_data_filepath must be a .pkl file.
        '''
        self.training_data_filepath = training_data_filepath
        self.training_data = pickle_load(training_data_filepath)
        self.to_fpobj_table = pickle_load(to_fpobj_table_filepath)
        self.training_mols = arrays_to_bitstrings(training_mols)
        self.test_mols = arrays_to_bitstrings(test_mols)
        
    def create_folds(self, pos_train_dict_filepath, unl_train_dict_filepath):
        train = pickle_load(pos_train_dict_filepath)
        unl_train = pickle_load(unl_train_dict_filepath)
        
        num_folds = 3
        self.num_folds = num_folds

        # create folds of molecules per enzyme
        train_folds = {}
        for ec in train:
            if len(ec.split('.')) < 4: # exclude internal EC levels
                continue

            mols = list(train[ec]['molecules']) + list(unl_train[ec]['molecules'])
            labels = list(np.ones(len(list(train[ec]['molecules'])))) + list(np.zeros(len(list(unl_train[ec]['molecules']))))
            num_mols = len(mols)
            temp = np.transpose(np.vstack((mols, labels)))
            np.random.shuffle(temp)
            mols = [int(x) for x in temp[:, 0]]
            labels = [float(x) for x in temp[:, 1]]

            train_folds[ec] = {'molecules': [], 'labels': []}
            for i in range(num_folds):
                if i < 4:
                    train_folds[ec]['molecules'].append(mols[i * (num_mols // num_folds) : (i + 1) * (num_mols // num_folds)])
                    train_folds[ec]['labels'].append(labels[i * (num_mols // num_folds) : (i + 1) * (num_mols // num_folds)])
                else:
                    train_folds[ec]['molecules'].append(mols[i * (num_mols // num_folds) :])
                    train_folds[ec]['labels'].append(labels[i * (num_mols // num_folds) :])
        
        return train_folds

    def optimize_for_K(self, pos_train_dict_filepath, unl_train_dict_filepath):
        train_folds = self.create_folds(pos_train_dict_filepath, unl_train_dict_filepath)
        
        # validation process
        K_list = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 'Mean']

        results = {}
        for enzyme in train_folds:
            print(enzyme)
            results[enzyme] = {}
            for i in range(self.num_folds):
                results[enzyme][i] = {}

                # make bitstrings for train and test data
                train_pos = []
                for j in range(self.num_folds):
                    if j != i:
                        train_pos += list(np.array(train_folds[enzyme]['molecules'][j])[np.array(train_folds[enzyme]['labels'][j]) == 1.0])

                test = train_folds[enzyme]['molecules'][i]
                y_true = train_folds[enzyme]['labels'][i]
                
                # map indices to molecules
                train_pos = self.training_mols[np.array(train_pos).astype(int)]
                test = self.training_mols[np.array(test).astype(int)] # still using training molecules!!!

                for K in K_list:
                    # compute similarities - need to convert each molecule to its fp representation
                    scores = []
                    truth = []
                    mols = []
                    for n, test_mol in enumerate(test):
                        test_fp = self.to_fpobj_table[test_mol]
                        if test_fp is None:
                            print('none')
                            continue
                        temp_scores = []
                        for pos_mol in train_pos:
                            pos_fp = self.to_fpobj_table[pos_mol]
                            if pos_fp is None:
                                print('none')
                                continue
                            temp_scores.append(DataStructs.FingerprintSimilarity(pos_fp, test_fp)) # FingerprintSimilarity

                        if len(temp_scores) == 0:
                            continue

                        if type(K) == int:
                            scores.append(np.mean(np.sort(temp_scores)[-K:]))
                        elif K == 'Mean':
                            scores.append(np.mean(temp_scores))
                        truth.append(y_true[n])

                        mols.append(test_mol)

                    try:
                        results[enzyme][i][K] = sklearn.metrics.average_precision_score(truth, scores)
                    except:
                        print('error')
                        results[enzyme][i][K] = np.NaN
        
        best_K_per_enzyme = {}
        for enzyme in results:
            scores_per_K = []
            for i in range(self.num_folds):
                scores_per_folds = []
                for K in K_list:
                    scores_per_folds.append(results[enzyme][i][K])
                scores_per_K.append(np.nanmean(scores_per_folds))
            best_K_per_enzyme[enzyme] = K_list[np.argmax(scores_per_K)]
        
        pickle_dump(best_K_per_enzyme, 'training_data/realistic_best_K_per_enzyme_morgan2binary.pkl')
    
    def test(self, pos_test_data_filepath, unl_test_data_filepath, EC=None):
        '''
        test_data_filepath must be a .pkl file.
        If EC is None, test() expects test_data_filepath to point to a dictionary where keys are EC Numbers
        and values are lists of molecules.
        Otherwise, if enzyme is an EC Number, test() expects test_data_filepath to point to a list of molecules.
        
        Inputs:

            test_data_filepath; string
                > .pkl file containing test data.

            EC; string (default None)
                > If None, test_data_filepath is expected to be a dict where keys are enzymes
                  and values are lists/arrays of molecules. If string and valid EC Number, 
                  test_data_filepath is expected to be a list of molecules to be tested
                  against EC.

        Outputs:

            results; dict
                > keys are enzymes and the values are dicts with keys: "Mean" (list of mean 
                similarity scores), "Max" (list of max similarity scores), "Molecules" (list 
                of molecules to which each score pertains).
        '''
        test_data = pickle_load(test_data_filepath)
        ###
        unl_test_dict = pickle_load(unl_test_data_filepath)
        best_K_per_enzyme = pickle_load('training_data/realistic_best_K_per_enzyme_morgan2binary.pkl')
        ###
        if EC is None: # dict case
            assert type(test_data) == dict
            
            enzymes = list(test_data.keys())
#             np.random.shuffle(enzymes)
        
            results = {}
            for enzyme in enzymes:
                if enzyme not in self.training_data:
                    continue
                
                if len(enzyme.split('.')) < 4:
                    continue
                
                print(enzyme)

                results[enzyme] = {}
                # make bitstrings for train and test data

                train_pos = self.training_data[enzyme]["molecules"]
                train_pos = self.training_mols[np.array(train_pos, dtype=int)]
                
                ###
                test = list(test_data[enzyme]["molecules"]) + list(unl_test_dict[enzyme]["molecules"])
                test = self.test_mols[np.array(test, dtype=int)]
                ###
                
                ###
                y_true = np.array(list(np.ones(len(test_data[enzyme]["molecules"]))) + list(np.zeros(len(unl_test_dict[enzyme]["molecules"]))))
                ###
                
                # compute similarities - need to convert each molecule to its fp representation
                mean_scores = []
                max_scores = []
                K_scores = []
                truth = []
                mols = []
                K = best_K_per_enzyme[enzyme]
                for i, test_mol in enumerate(test):
                    test_fp = self.to_fpobj_table[test_mol]
                    if test_fp is None:
                        continue
                    temp_scores = []
                    for pos_mol in train_pos:
                        pos_fp = self.to_fpobj_table[pos_mol]
                        if pos_fp is None:
                            continue
                        temp_scores.append(DataStructs.FingerprintSimilarity(pos_fp, test_fp)) # FingerprintSimilarity

                    if len(temp_scores) == 0:
                        continue
                    mean_scores.append(np.mean(temp_scores))
                    max_scores.append(np.max(temp_scores))
                    if K == 'Mean':
                        K_scores.append(np.mean(temp_scores))
                    else:
                        K_scores.append(np.mean(np.sort(temp_scores)[-K:]))
                    ###
                    truth.append(y_true[i])
                    ###
                    mols.append(test_mol)
                
                results[enzyme]["Truth"] = truth
                results[enzyme]["Molecules"] = mols
                results[enzyme]["Predictions"] = K_scores
                
        else: # individual enzyme case
            assert type(enzyme) == str
            if enzyme not in self.training_data:
                raise Exception(enzyme + " is not in training data.")
            
            results = {enzyme: {}}
                # make bitstrings for train and test data

                train_pos = self.training_data[enzyme]["molecules"]
                train_pos = self.training_mols[np.array(train_pos, dtype=int)]
                
                ###
                test = list(test_data[enzyme]["molecules"]) + list(unl_test_dict[enzyme]["molecules"])
                test = self.test_mols[np.array(test, dtype=int)]
                ###
                
                ###
                y_true = np.array(list(np.ones(len(test_data[enzyme]["molecules"]))) + list(np.zeros(len(unl_test_dict[enzyme]["molecules"]))))
                ###

            # compute similarities - need to convert each molecule to its fp representation
            mean_scores = []
            max_scores = []
            K_scores = []
            truth = []
            mols = []
            K = best_K_per_enzyme[enzyme]
            for i, test_mol in enumerate(test):
                test_fp = self.to_fpobj_table[test_mol]
                if test_fp is None:
                    continue
                temp_scores = []
                for pos_mol in train_pos:
                    pos_fp = self.to_fpobj_table[pos_mol]
                    if pos_fp is None:
                        continue
                    temp_scores.append(DataStructs.FingerprintSimilarity(pos_fp, test_fp)) # FingerprintSimilarity

                if len(temp_scores) == 0:
                    continue
                mean_scores.append(np.mean(temp_scores))
                max_scores.append(np.max(temp_scores))
                if K == 'Mean':
                    K_scores.append(np.mean(temp_scores))
                else:
                    K_scores.append(np.mean(np.sort(temp_scores)[-K:]))
                ###
                truth.append(y_true[i])
                ###
                mols.append(test_mol)

            results[enzyme]["Truth"] = truth
            results[enzyme]["Molecules"] = mols
            results[enzyme]["Predictions"] = K_scores      
            
        return results