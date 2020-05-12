import numpy as np
import argparse
from utils import pickle_load, pickle_dump, array_to_bit_string

TEST_RATIO = 0.20

def process_from_hmcnf_to_tree_classifier(data, labels, sim_weights, bal_weights, bal_sim_weights, enzyme_index_dict):
    pos_dict = {}
    unl_dict = {}
    for enzyme_i in range(labels.shape[1]):
        pos_dict[enzyme_index_dict[enzyme_i]] = {}
        pos_dict[enzyme_index_dict[enzyme_i]]["molecules"] = []
        pos_dict[enzyme_index_dict[enzyme_i]]["sim_weights"] = []
        pos_dict[enzyme_index_dict[enzyme_i]]["bal_weights"] = []
        pos_dict[enzyme_index_dict[enzyme_i]]["bal_sim_weights"] = []
        unl_dict[enzyme_index_dict[enzyme_i]] = {}
        unl_dict[enzyme_index_dict[enzyme_i]]["molecules"] = []
        unl_dict[enzyme_index_dict[enzyme_i]]["sim_weights"] = []
        unl_dict[enzyme_index_dict[enzyme_i]]["bal_weights"] = []
        unl_dict[enzyme_index_dict[enzyme_i]]["bal_sim_weights"] = []
        for mol_i in range(labels.shape[0]):
            if labels[mol_i][enzyme_i] == 1.0:
                pos_dict[enzyme_index_dict[enzyme_i]]["molecules"].append(array_to_bit_string(data[mol_i]))
                pos_dict[enzyme_index_dict[enzyme_i]]["sim_weights"].append(sim_weights[mol_i][enzyme_i])
                pos_dict[enzyme_index_dict[enzyme_i]]["bal_weights"].append(bal_weights[mol_i][enzyme_i])
                pos_dict[enzyme_index_dict[enzyme_i]]["bal_sim_weights"].append(bal_sim_weights[mol_i][enzyme_i])
            elif labels[mol_i][enzyme_i] == 0.0:
                unl_dict[enzyme_index_dict[enzyme_i]]["molecules"].append(array_to_bit_string(data[mol_i]))
                unl_dict[enzyme_index_dict[enzyme_i]]["sim_weights"].append(sim_weights[mol_i][enzyme_i])
                unl_dict[enzyme_index_dict[enzyme_i]]["bal_weights"].append(bal_weights[mol_i][enzyme_i])
                unl_dict[enzyme_index_dict[enzyme_i]]["bal_sim_weights"].append(bal_sim_weights[mol_i][enzyme_i])
            else:
                print("labels[enzyme_i][mol_i] should be 0.0 or 1.0, but it is %.3f" % (labels[enzyme_i][mol_i]))
    
    return pos_dict, unl_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inhibitors', default="False")
    args = parser.parse_args()

    if args.inhibitors.lower() == "true":
        inh = "_inh"
    elif args.inhibitors.lower() == "false":
        inh = ""
    else:
        print("Argument Error: --inhibitors must be given a valid boolean identifier.")
        exit(1)

    data = pickle_load("../data/HMCNF_data/data.pkl")
    Pl1 = pickle_load("../data/HMCNF_data/Pl1.pkl")
    Pl2 = pickle_load("../data/HMCNF_data/Pl2.pkl")
    Pl3 = pickle_load("../data/HMCNF_data/Pl3.pkl")
    Pl4 = pickle_load("../data/HMCNF_data/Pl4.pkl")

    Pl1_sim_weights = pickle_load("../data/HMCNF_data/Pl1_sim_weights%s.pkl" % (inh))
    Pl2_sim_weights = pickle_load("../data/HMCNF_data/Pl2_sim_weights%s.pkl" % (inh))
    Pl3_sim_weights = pickle_load("../data/HMCNF_data/Pl3_sim_weights%s.pkl" % (inh))
    Pl4_sim_weights = pickle_load("../data/HMCNF_data/Pl4_sim_weights%s.pkl" % (inh))

    Pl1_sim_bal_weights = pickle_load("../data/HMCNF_data/Pl1_sim_bal_weights%s.pkl" % (inh))
    Pl2_sim_bal_weights = pickle_load("../data/HMCNF_data/Pl2_sim_bal_weights%s.pkl" % (inh))
    Pl3_sim_bal_weights = pickle_load("../data/HMCNF_data/Pl3_sim_bal_weights%s.pkl" % (inh))
    Pl4_sim_bal_weights = pickle_load("../data/HMCNF_data/Pl4_sim_bal_weights%s.pkl" % (inh))

    Pl1_bal_weights = pickle_load("../data/HMCNF_data/Pl1_bal_weights.pkl")
    Pl2_bal_weights = pickle_load("../data/HMCNF_data/Pl2_bal_weights.pkl")
    Pl3_bal_weights = pickle_load("../data/HMCNF_data/Pl3_bal_weights.pkl")
    Pl4_bal_weights = pickle_load("../data/HMCNF_data/Pl4_bal_weights.pkl")

    indexes = pickle_load("../utils/indexes_for_splitting.pkl")
    ec_index_dict = pickle_load("../utils/ec_index_dict.pkl")
    class_index_dict = pickle_load("../utils/class_index_dict.pkl")
    subclass_index_dict = pickle_load("../utils/subclass_index_dict.pkl")
    subsubclass_index_dict = pickle_load("../utils/subsubclass_index_dict.pkl")

    cutoff = int(data.shape[0]*TEST_RATIO)

    train_i = indexes[cutoff:]
    test_i = indexes[:cutoff]

    EC_pos_dict, EC_unl_dict = process_from_hmcnf_to_tree_classifier(data[train_i], Pl4[train_i], Pl4_sim_weights[train_i], Pl4_bal_weights[train_i], Pl4_sim_bal_weights[train_i], ec_index_dict)
    print("train ec done")
    class_pos_dict, class_unl_dict = process_from_hmcnf_to_tree_classifier(data[train_i], Pl1[train_i], Pl1_sim_weights[train_i], Pl1_bal_weights[train_i], Pl1_sim_bal_weights[train_i], class_index_dict)
    print("train class done")
    subclass_pos_dict, subclass_unl_dict = process_from_hmcnf_to_tree_classifier(data[train_i], Pl2[train_i], Pl2_sim_weights[train_i], Pl2_bal_weights[train_i], Pl2_sim_bal_weights[train_i], subclass_index_dict)
    print("train subclass done")
    subsubclass_pos_dict, subsubclass_unl_dict = process_from_hmcnf_to_tree_classifier(data[train_i], Pl3[train_i], Pl3_sim_weights[train_i], Pl3_bal_weights[train_i], Pl3_sim_bal_weights[train_i], subsubclass_index_dict)
    print("train subsubclass done")

    EC_pos_dict_test, EC_unl_dict_test = process_from_hmcnf_to_tree_classifier(data[test_i], Pl4[test_i], Pl4_sim_weights[test_i], Pl4_bal_weights[test_i], Pl4_sim_bal_weights[test_i], ec_index_dict)
    print("test ec done")
    class_pos_dict_test, class_unl_dict_test = process_from_hmcnf_to_tree_classifier(data[test_i], Pl1[test_i], Pl1_sim_weights[test_i], Pl1_bal_weights[test_i], Pl1_sim_bal_weights[test_i], class_index_dict)
    print("test class done")
    subclass_pos_dict_test, subclass_unl_dict_test = process_from_hmcnf_to_tree_classifier(data[test_i], Pl2[test_i], Pl2_sim_weights[test_i], Pl2_bal_weights[test_i], Pl2_sim_bal_weights[test_i], subclass_index_dict)
    print("test subclass done")
    subsubclass_pos_dict_test, subsubclass_unl_dict_test = process_from_hmcnf_to_tree_classifier(data[test_i], Pl3[test_i], Pl3_sim_weights[test_i], Pl3_bal_weights[test_i], Pl3_sim_bal_weights[test_i], subsubclass_index_dict)
    print("test subsubclass done")

    pos_dict = {**EC_pos_dict, **class_pos_dict, **subclass_pos_dict, **subsubclass_pos_dict}
    unl_dict = {**EC_unl_dict, **class_unl_dict, **subclass_unl_dict, **subsubclass_unl_dict}
    pickle_dump(pos_dict, "../data/tree_data/pos_dict_train%s.pkl" % inh)
    pickle_dump(unl_dict, "../data/tree_data/unl_dict_train%s.pkl" % inh)

    pos_dict = {**EC_pos_dict_test, **class_pos_dict_test, **subclass_pos_dict_test, **subsubclass_pos_dict_test}
    unl_dict = {**EC_unl_dict_test, **class_unl_dict_test, **subclass_unl_dict_test, **subsubclass_unl_dict_test}
    pickle_dump(pos_dict, "../data/tree_data/pos_dict_test%s.pkl" % inh)
    pickle_dump(unl_dict, "../data/tree_data/unl_dict_test%s.pkl" % inh)