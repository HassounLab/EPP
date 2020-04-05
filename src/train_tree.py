
import sys
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import time
import pickle
from scipy import interp

from TreeModelWithCVWeightedWithInhibitorsAllNeg_no_folds import EnzymePredictionTreeModelWithCVWeightedWithInhibitorsAllNeg

def dump(data, filename):
    f = open(filename, "wb")
    pickle.dump(data, f)
    f.close()

def pickle_load(filename):
    f = open(filename, "rb")
    data = pickle.load(f)
    f.close()
    return data

######## functions for testing ########

def get_pr_curves(model, est, weighted):
    first_level, second_level, third_level, fourth_level = get_list_of_enzymes(model)
    PRs = {}
    
    pos_preds = []
    neg_preds = []
    for enzyme in fourth_level:
        ec_breakdown = enzyme.split('.')
        depth = 0
        currpos = model.model_tree
        while depth < len(ec_breakdown):
            if ec_breakdown[depth] not in currpos:
                print("Enzyme not present in the Model at depth %d." % (depth + 1))
                return None
            elif currpos[ec_breakdown[depth]]["Estimator"] == None:
                print("Enzyme has less than %d Positive examples at depth %d." % (self.min_positive_examples, depth + 1))
                return None
            currpos = currpos[ec_breakdown[depth]]
            depth += 1
        
        x_test = currpos["X Test"]
        y_test = currpos["Y Test"]
#         w_test_N_list = currpos["W Test"]
        # add_inh = False
        # if "Inh Test" in currpos:
        #     x_inh_list = currpos["Inh Test"]
        #     if x_inh_list[0].shape[0] != 0:
        #         add_inh = True
        
        PRs[enzyme] = {}
        PRs[enzyme]["num_pos_train"] = currpos["# Positive Examples"]
        PRs[enzyme]["num_pos_test"] = currpos["Y Test"][currpos["Y Test"] == 1.0].shape[0]

        y_hat = model.predict_proba_for_testing(x_test, enzyme, est)
        pos_preds += list(y_hat[y_test == 1.0])
        neg_preds += list(y_hat[y_test == 0.0])
#             print(y_hat)
        
        if weighted:
            prec, rec, thr = sklearn.metrics.precision_recall_curve(y_test, y_hat, sample_weight=w_test)
            ap = sklearn.metrics.average_precision_score(y_test, y_hat, sample_weight=w_test)
        else:
            prec, rec, thr = sklearn.metrics.precision_recall_curve(y_test, y_hat)
            ap = sklearn.metrics.average_precision_score(y_test, y_hat)
            
        rec = rec[::-1]     # why do I do this?? for interp purposes maybe?
        prec = prec[::-1]
            
        PRs[enzyme]["prec"] = prec
        PRs[enzyme]["rec"] = rec
        PRs[enzyme]["ap"] = ap
        PRs[enzyme]["y_hat"] = y_hat
        PRs[enzyme]["y_true"] = y_test
        
    return PRs, pos_preds, neg_preds

def get_rocs(model, est, weighted):
    first_level, second_level, third_level, fourth_level = get_list_of_enzymes(model)
    rocs = {}
    for enzyme in fourth_level:
        ec_breakdown = enzyme.split('.')
        depth = 0
        currpos = model.model_tree
        while depth < len(ec_breakdown):
            if ec_breakdown[depth] not in currpos:
                print("Enzyme not present in the Model at depth %d." % (depth + 1))
                return None
            elif currpos[ec_breakdown[depth]]["Estimator"] == None:
                print("Enzyme has less than %d Positive examples at depth %d." % (self.min_positive_examples, depth + 1))
                return None
            currpos = currpos[ec_breakdown[depth]]
            depth += 1
        
        x_test = currpos["X Test"]
        y_test = currpos["Y Test"]
#         w_test_N_list = currpos["W Test"]
        # add_inh = False
        # if "Inh Test" in currpos:
        #     x_inh_list = currpos["Inh Test"]
        #     if x_inh_list[0].shape[0] != 0:
        #         add_inh = True
        
        rocs[enzyme] = {}

        # x_test = x_test_NF_list[i]
        # y_test = y_test_N_list[i]

        y_hat = model.predict_proba_for_testing(x_test, enzyme, est)
        
        if weighted:
            fpr, tpr, thr = sklearn.metrics.roc_curve(y_test, y_hat, sample_weight=w_test_N_list[i])
        else:
            fpr, tpr, thr = sklearn.metrics.roc_curve(y_test, y_hat)
            
        rocs[enzyme]["tpr"] = tpr
        rocs[enzyme]["fpr"] = fpr
    return rocs

def get_list_of_enzymes(model):
    for ec in model.enzymes_tree:
        if model.model_tree[ec]["Estimator"] is None:
            continue
        model.model_tree[ec]["Enzyme"] = ec
        for x in model.enzymes_tree[ec]:
            if model.model_tree[ec][x]["Estimator"] is None:
                continue
            model.model_tree[ec][x]["Enzyme"] = ec+"."+x
            for y in model.enzymes_tree[ec][x]:
                if model.model_tree[ec][x][y]["Estimator"] is None:
                    continue
                model.model_tree[ec][x][y]["Enzyme"] = ec+"."+x+"."+y
                for z in model.enzymes_tree[ec][x][y]:
                    if model.model_tree[ec][x][y][z]["Estimator"] is None:
                        continue
                    model.model_tree[ec][x][y][z]["Enzyme"] = ec+"."+x+"."+y+"."+z
    
    first_level = []
    second_level = []
    third_level = []
    fourth_level = []
    depth = 0
    for ec in model.enzymes_tree:
        if model.model_tree[ec]["Estimator"] == None:
            continue
        first_level.append(model.model_tree[ec]["Enzyme"])
        for x in model.enzymes_tree[ec]:
            if model.model_tree[ec][x]["Estimator"] == None:
                continue
            second_level.append(model.model_tree[ec][x]["Enzyme"])
            for y in model.enzymes_tree[ec][x]:
                if model.model_tree[ec][x][y]["Estimator"] == None:
                    continue
                third_level.append(model.model_tree[ec][x][y]["Enzyme"])
                for z in model.enzymes_tree[ec][x][y]:
                    if model.model_tree[ec][x][y][z]["Estimator"] == None:
                        continue
                    fourth_level.append(model.model_tree[ec][x][y][z]["Enzyme"])
    return first_level, second_level, third_level, fourth_level

######## functions for testing #########

if __name__ == "__main__":
    r = "all"
    train_filepath = "data_tree/all_data_all_unl_all_levels_inh_train_r%s.pkl" % (r)
    test_filepath = "data_tree/all_data_all_unl_all_levels_inh_test_r%s.pkl" % (r)

    print("Creating Tree...")
    from TreeModelWithCVWeightedWithInhibitorsAllNeg_no_folds import EnzymePredictionTreeModelWithCVWeightedWithInhibitorsAllNeg
    model = EnzymePredictionTreeModelWithCVWeightedWithInhibitorsAllNeg(train_filepath = train_filepath, test_filepath = test_filepath, min_positive_examples = 10)
    model.create_ec_numbers_tree()
    model.create_tree()
    dump(model, "tree_results/tree_classifier_inh_r%s.pkl" % (r))
    # model = pickle_load("tree_results/all_neg_with_inh")
    # print(model.enzymes_tree)

    # print("Getting PR curves")
    # # PRs_w_h, pos_preds, neg_preds = get_pr_curves(model, "Estimator", False)
    # PRs_w, pos_preds, neg_preds = get_pr_curves(model, "Est Ref RFClas", False)
    # # PRs_h, pos_preds, neg_preds = get_pr_curves(model, "Est No Weight", False)
    # PRs, pos_preds, neg_preds = get_pr_curves(model, "Est Ref RFClas No Weight", False)

    # print("Getting ROCs")
    # # rocs_w_h = get_rocs(model, "Estimator", False)
    # rocs_w = get_rocs(model, "Est Ref RFClas", False)
    # # rocs_h = get_rocs(model, "Est No Weight", False)
    # rocs = get_rocs(model, "Est Ref RFClas No Weight", False)   

    # print("Dumping")
    # # dump(PRs_w_h, "tree_results/PRs_w_h_all_neg_no_folds.pkl")
    # dump(PRs_w, "tree_results/PRs_w_all_neg_no_folds.pkl")
    # # dump(PRs_h, "tree_results/PRs_h_all_neg_no_folds.pkl")
    # dump(PRs, "tree_results/PRs_all_neg_no_folds.pkl")

    # # dump(rocs_w_h, "tree_results/rocs_w_h_all_neg_no_folds.pkl")
    # dump(rocs_w, "tree_results/rocs_w_all_neg_no_folds.pkl")
    # # dump(rocs_h, "tree_results/rocs_h_all_neg_no_folds.pkl")
    # dump(rocs, "tree_results/rocs_all_neg_no_folds.pkl")
    # print("done")