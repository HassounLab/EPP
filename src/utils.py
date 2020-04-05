'''
General Utilities for all models

'''
import pickle
import numpy as np
from numpy import interp


def pickle_load(filepath):
    f = open(filepath, "rb")
    data = pickle.load(f)
    f.close()
    return data

def pickle_dump(data, filepath):
    f = open(filepath, "wb")
    pickle.dump(data, f)
    f.close()

def get_data(filepath):
    data = dict()
    handle = open(filepath, 'r')
    for line in handle:
        line = line.split('\t')
        ecnumber = line[0]
        smiles = line[1].strip('\n').split('$')
        data[ecnumber] = smiles
    handle.close()
    return data

def write_data(self, data, filepath):
    handle = open(filepath, "w+")

    for enzyme in data:
        handle.write(enzyme)
        handle.write('\t')
        handle.write(self.concatenate(data[enzyme], '$'))
        handle.write('\n')

    handle.close()

def bit_strings_to_arrays(bit_strings):
    bit_arrays = []
    for bit_string in bit_strings:
        bit_arrays.append(np.array([int(i) for i in bit_string]))
    return np.array(bit_arrays)

def remove_duplicates(alist):
    newlist = []
    for elem in alist:
        if elem not in newlist:
            newlist.append(elem)
    return newlist

def polyfit(x, y, degree):
    results = {"Coefficients": [], "R-squared": 0}

    coeffs = np.polyfit(x, y, degree)

    results["Coefficients"] = coeffs.tolist()

    # R-squared calculation
    p = np.poly1d(coeffs)
    y_fit = p(x)                        
    y_mean = np.sum(y)/len(y)
    ssreg = np.sum((y_fit-y_mean)**2)
    sstot = np.sum((y - y_mean)**2)
    
    results["R-squared"] = ssreg / sstot

    return results