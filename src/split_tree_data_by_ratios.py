'''
DEPRECATED
'''

import numpy as np
import scipy
import sklearn
import sklearn.metrics
import argparse

from utils import pickle_load, pickle_dump


def get_random_sample(num_sample, max_sample):
    sample = np.arange(0, max_sample, dtype=int)
    np.random.shuffle(sample)

    return sample[:num_sample]

def get_unl_for_ratio(pos_dict, unl_dict, r):
    new_unl_dict = {}

    for enzyme in unl_dict:
        num_pos = len(pos_dict[enzyme]["molecules"])
        if num_pos == 0:   # numerically stable
            continue
        num_unl = len(unl_dict[enzyme]["molecules"])
        if r == "all":
            sample_num_unl = num_unl
        else:
            r = float(r)
            sample_num_unl = min(max(int(num_pos * (1 - r)/r), 1), num_unl)   # numerically stable
        sample_unl = get_random_sample(sample_num_unl, num_unl)

        new_unl_dict[enzyme] = {}
        new_unl_dict[enzyme]["molecules"] = np.array(unl_dict[enzyme]["molecules"])[sample_unl]
        new_unl_dict[enzyme]["sim_weights"] = np.array(unl_dict[enzyme]["sim_weights"])[sample_unl]

        # balancing weights
        new_unl_dict[enzyme]["bal_weights"] = np.full(sample_num_unl, float(num_pos)/float(sample_num_unl))

        # similarity-biased balancing weights
        sum_sim_weights = np.sum(new_unl_dict[enzyme]["sim_weights"])
        balancing_factor = float(num_pos)/sum_sim_weights
        new_unl_dict[enzyme]["bal_sim_weights"] = np.copy(new_unl_dict[enzyme]["sim_weights"])*(float(num_pos)/sum_sim_weights)

    return new_unl_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inhibitors', default="False")
    parser.add_argument('--r_list', default='[all, 0.01, 0.05, 0.20, 0.50]')
    args = parser.parse_args()

    if args.inhibitors.lower() == "true":
        inh = "_inh"
    elif args.inhibitors.lower() == "false":
        inh = ""
    else:
        print("Argument Error: --inhibitors must be given a valid boolean identifier.")
        exit(1)

    r_list = args.r_list.strip(' ').strip('[').strip(']').split(',') # convert from string to list of strings

    pos_dict_train = pickle_load("../data/tree_data/pos_dict_train%s.pkl" % (inh))
    unl_dict_train = pickle_load("../data/tree_data/unl_dict_train%s.pkl" % (inh))

    for r in r_list:
        r = r.strip(' ')
        r_unl_dict_train = get_unl_for_ratio(pos_dict_train, unl_dict_train, r)
        pickle_dump(r_unl_dict_train, "../data/tree_data/unl_dict_train%s_r%s.pkl" % (inh, r))
