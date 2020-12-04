import pickle
import numpy as np
from rdkit.Chem import MolFromSmiles, MACCSkeys, AllChem
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import pandas as pd
import argparse

def pickle_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def pickle_dump(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(f, data)

def bitstring_to_array(bitstring):
    return np.array([float(bit) for bit in bitstring])

def read_input_smiles(smilesfile):
    molecules = []
    with open(smilesfile, 'r') as f:
        for line in f:
            molecules.append(line.strip('\n'))

    return molecules

def convert_to_maccs(smiles):
    smiles_maccs_table = {}
    for smile in smiles:
        if smile in smiles_maccs_table:
            continue
        mol = MolFromSmiles(smile)
        if mol is None: # could not convert smile to Mol object
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        if fp is None: # could not generate maccs fingerprint from Mol object
            continue
        smiles_maccs_table[smile] = bitstring_to_array(fp.ToBitString())

    filtered_smiles = np.array(smiles_maccs_table.keys())
    maccs = np.array(list(smiles_maccs_table.values()))
    return filtered_smiles, maccs, smiles_maccs_table

def convert_to_morgan2(smiles):
    smiles_morgan2_table = {}
    for smile in smiles:
        if smile in smiles_morgan2_table: # duplicate smile
            continue
        mol = MolFromSmiles(smile)
        if mol is None: # could not convert smile to Mol object
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        if fp is None: # could not generate morgan2 fingerprint from Mol object
            continue
        smiles_morgan2_table[smile] = bitstring_to_array(fp.ToBitString())

    filtered_smiles, morgan2 = [[i for i, j in smiles_morgan2_table.items()], [j for i, j in smiles_morgan2_table.items()]]
    return np.array(filtered_smiles), np.array(morgan2), smiles_morgan2_table

def is_enzyme_in_model(enzyme):
    return enzyme in EC_INDEX_DICT

def test_on_molecules(model, molecules, beta=0.5):
    Pg, Pl1, Pl2, Pl3, Pl4 = model(molecules.float(), training=False)
    predictions = beta*torch.cat((Pl4, Pl3, Pl2, Pl1), dim=1) + (1 - beta)*Pg
    return predictions.detach().numpy()

def filter_predictions(predictions, enzymes):
    '''
    returns predictions_dict: keys are EC Numbers, values are arrays with predictions
    '''

    predictions_dict = {}
    for enzyme in enzymes:
        try:
            ec_i = EC_INDEX_DICT[enzyme]
        except KeyError: # if enzyme is not in our model
            pass

        predictions_dict[enzyme] = predictions[:, ec_i]

    return predictions_dict

################################ Model Code #######################################

x_size = 2048

class Net(nn.Module):

    def __init__(self, C, Cl1, Cl2, Cl3, Cl4, dropout, h_size, reduced=False):
        super(Net, self).__init__()
        self.C = C     # total number of classes
        self.Cl1 = Cl1
        self.Cl2 = Cl2
        self.Cl3 = Cl3
        self.Cl4 = Cl4
        self.reduced = reduced
        self.dropout = dropout
        self.h_size = h_size
        
        self.global1 = nn.Linear(x_size, h_size)
        self.batch_norm1 = nn.BatchNorm1d(h_size)
        self.global2 = nn.Linear(h_size + x_size, h_size)
        self.batch_norm2 = nn.BatchNorm1d(h_size)
        if not self.reduced:
            self.global3 = nn.Linear(h_size + x_size, h_size)
        else:
            self.global3 = nn.Linear(x_size, h_size)
        self.batch_norm3 = nn.BatchNorm1d(h_size)
        self.global4 = nn.Linear(h_size + x_size, h_size)
        self.batch_norm4 = nn.BatchNorm1d(h_size)
        self.globalOut = nn.Linear(h_size, self.C)
        
        self.local1 = nn.Linear(h_size, h_size)
        self.batch_normL1 = nn.BatchNorm1d(h_size)
        self.localOut1 = nn.Linear(h_size, self.Cl1)
        self.local2 = nn.Linear(h_size, h_size)
        self.batch_normL2 = nn.BatchNorm1d(h_size)
        self.localOut2 = nn.Linear(h_size, self.Cl2)
        self.local3 = nn.Linear(h_size, h_size)
        self.batch_normL3 = nn.BatchNorm1d(h_size)
        self.localOut3 = nn.Linear(h_size, self.Cl3)
        self.local4 = nn.Linear(h_size, h_size)
        self.batch_normL4 = nn.BatchNorm1d(h_size)
        self.localOut4 = nn.Linear(h_size, self.Cl4)

    def forward(self, x, training=True):
        if not self.reduced:
            Ag1 = F.dropout(self.batch_norm1(F.relu(self.global1(x))), p=self.dropout, training=training)
            Al1 = F.dropout(self.batch_normL1(F.relu(self.local1(Ag1))), p=self.dropout, training=training)
            Pl1 = torch.sigmoid(self.localOut1(Al1))

            Ag2 = F.dropout(self.batch_norm2(F.relu(self.global2(torch.cat([Ag1, x], dim=1)))), p=self.dropout, training=training)
            Al2 = F.dropout(self.batch_normL2(F.relu(self.local2(Ag2))), p=self.dropout, training=training)
            Pl2 = torch.sigmoid(self.localOut2(Al2))
        
            Ag3 = F.dropout(self.batch_norm3(F.relu(self.global3(torch.cat([Ag2, x], dim=1)))), p=self.dropout, training=training)
        else:
            Ag3 = F.dropout(self.batch_norm3(F.relu(self.global3(x))), p=self.dropout, training=training)
        
        Al3 = F.dropout(self.batch_normL3(F.relu(self.local3(Ag3))), p=self.dropout, training=training)
        Pl3 = torch.sigmoid(self.localOut3(Al3))
        
        Ag4 = F.dropout(self.batch_norm4(F.relu(self.global4(torch.cat([Ag3, x], dim=1)))), p=self.dropout, training=training)
        Al4 = F.dropout(self.batch_normL4(F.relu(self.local4(Ag4))), p=self.dropout, training=training)
        Pl4 = torch.sigmoid(self.localOut4(Al4))
        
        Pg = torch.sigmoid(self.globalOut(Ag4))
        
        if not self.reduced:
            return Pg, Pl1, Pl2, Pl3, Pl4    # return all outputs to compute loss
        else:
            return Pg, Pl3, Pl4

################################ Model Code #######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smilesfile', default='testsmiles.txt')
    parser.add_argument('--outputfile', default='output.csv')
    args = parser.parse_args()
    smilesfile = args.smilesfile
    outputfile = args.outputfile

    EC_INDEX_DICT = pickle_load("ec_index_dict.pkl")
    ALL_ENZYMES = filter(lambda x : type(x) == str, list(EC_INDEX_DICT.keys()))

    '''
    For some reason, pytorch gives a warning that the source code of the model has changed, but I simply copy-pasted the code.
    The results it gives are consistent with the expectations, so I think we can just silence the warning?
    '''
    warnings.filterwarnings("ignore")
    model = torch.load("HMCNF_morgan2binary_inh.pt")
    device = torch.device("cpu")

    # this is a simple example. all these molecules come from EC 1.1.1.149, and thus they should all have high scores for 1.1.1.149 and probably
    # very low scores for 6.2.1.12 (I chose it arbitrarily)
    # molecules = ["CC(C1(CCC2C1(CCC3C2CCC4=CC(=O)CCC34C)C)O)O", "fake", "CC(=O)C1(CCC2C1(CCC3C2CCC4=CC(=O)CCC34C)C)O","CC(C1CCC2C1(CCC3C2CCC4=CC(=O)CCC34C)C)O","CC(=O)C1CCC2C1(CCC3C2CCC4=CC(=O)CCC34C)C","CC(C1CCC2C1(CCC3C2CCC4C3(CCC(=O)C4)C)C)O"]
    # enzymes = ["1.1.1.149", "6.2.1.12"]

    enzymes = ALL_ENZYMES
    molecules = read_input_smiles(smilesfile)

    ############################## predictions procedure ##########################################
    
    smiles, morgan2, smiles_morgan2_table = convert_to_morgan2(molecules)

    if smiles_morgan2_table == {}:
        print("Could not convert any of the given smiles to morgan2 fingerprints.")
        exit(1)

    predictions = test_on_molecules(model, torch.from_numpy(morgan2))

    predictions_dict = filter_predictions(predictions, enzymes)

    if len(predictions_dict) == 0:
        print("None of the queried enzymes are present in our model.")
        exit(1)

    ############################## predictions procedure ##########################################

    predictions_dict['smiles'] = smiles
    df = pd.DataFrame(predictions_dict)
    df.set_index('smiles', inplace=True)
    df.to_csv(outputfile)
