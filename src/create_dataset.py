import numpy as np
import pickle
import sys

from utils import pickle_load, pickle_dump, get_data, remove_duplicates, bit_strings_to_arrays

class dataset(object):
    def __init__(self, filepath_pos_maccs, filepath_unl_maccs, num_folds=5):
        self.pos_bitstrings = pickle_load(filepath_pos_maccs)
        self.unl_bitstrings = pickle_load(filepath_unl_maccs)
        self.num_folds=num_folds
        self.data_ready = {}
        
    def prepare_data(self, filepath=None):
        for enzyme in self.pos_bitstrings:
            if len(self.pos_bitstrings[enzyme]["molecules"]) > 0:
                print("Doing enzyme: {}".format(enzyme))
                x_NF, y_N, num_pos, sample_weights_bal, sample_weights_sim, inh_test = self.prepare_data_single(enzyme)
            else:
                print("Doing enzyme: {} --> failed".format(enzyme))
                continue

            if inh_test is None:
                has_inh = False
            else:
                has_inh = True

            self.data_ready[enzyme] = [x_NF, y_N, num_pos, sample_weights_bal, sample_weights_sim, inh_test]
        
        if filepath is not None:
            pickle_dump(self.data_ready, filepath)

    def prepare_data_single(self, enzyme):
        '''
        Gets and prepares data for the purpose of using it in machine learning models.
        Returns:
            data examples
            target examples
            testing data examples (if in testing mode)
            testing target examples (if in testing mode)
            number of positive examples (number of negative is made to be the same as the positive)
        '''

        # First get positive data
        maccs_pos = self.pos_bitstrings[enzyme]["molecules"]

        # Then get negative data
        unl_data_maccs = bit_strings_to_arrays(self.unl_bitstrings[enzyme]["molecules"])
        unl_bal_weights = self.unl_bitstrings[enzyme]["bal_weights"]
        unl_sim_bal_weights = self.unl_bitstrings[enzyme]["bal_sim_weights"]

        # Convert bit strings to arrays of binary
        pos_data_maccs = bit_strings_to_arrays(maccs_pos)
        pos_bal_weights = self.pos_bitstrings[enzyme]["bal_weights"]
        pos_sim_bal_weights = self.pos_bitstrings[enzyme]["bal_sim_weights"]

        # Now prepare data to be used for machine learning purposes
        y_pos = np.full(pos_data_maccs.shape[0], 1.0)
        y_unl = np.full(unl_data_maccs.shape[0], 0.0)

        pos_data = np.vstack((np.transpose(pos_data_maccs), y_pos, pos_bal_weights, pos_sim_bal_weights))
        pos_data = np.transpose(pos_data)

        neg_data = np.vstack((np.transpose(unl_data_maccs), y_unl, unl_bal_weights, unl_sim_bal_weights))
        neg_data = np.transpose(neg_data)

        all_data = np.vstack((pos_data, neg_data))
        np.random.shuffle(all_data)

        x_NF = all_data[:,:-3]
        y_N = all_data[:,-3]
        sample_weights_bal = all_data[:,-2]
        sample_weights_sim_bal = all_data[:,-1]

        return x_NF, y_N, int(pos_data.shape[0]), sample_weights_bal, sample_weights_sim_bal, inh_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inhibitors', default=False)
    parser.add_argument('--ratio', default="all")
    args = parser.parse_args()

    inh = "_inh" if args.inhibitors else ""
    r = args.ratio

    print("Doing training")
    dset = dataset("../data/pos_dict_train.pkl" % (inh), "../data/unl_dict_train%s_r%s.pkl" % (inh, r))
    data_filepath = "../data/tree_data/tree_and_flat_data%s_train_r%s.pkl" % (inh, r)
    dset.prepare_data(filepath=data_filepath)

    print("Doing testing")
    dset_test = dataset("../data/pos_dict_test.pkl" % (inh), "../data/unl_dict_test%s_r%s.pkl" % (inh, r))
    data_filepath_test = "../data/tree_data/tree_and_flat_data%s_test_r%s.pkl" % (inh, r)
    dset_test.prepare_data(filepath=data_filepath_test)
    print("Done")
