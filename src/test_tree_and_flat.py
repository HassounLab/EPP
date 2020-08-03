import numpy as np
import argparse

from utils import pickle_load, pickle_dump
from TreeAndFlatModels import TreeAndFlatModels

from utils import pickle_load, pickle_dump

def get_list_of_enzymes(model):
    for ec in model.enzymes_tree:
        if model.model_tree[ec]["Tree"] is None:
            continue
        model.model_tree[ec]["Enzyme"] = ec
        for x in model.enzymes_tree[ec]:
            if model.model_tree[ec][x]["Tree"] is None:
                continue
            model.model_tree[ec][x]["Enzyme"] = ec+"."+x
            for y in model.enzymes_tree[ec][x]:
                if model.model_tree[ec][x][y]["Tree"] is None:
                    continue
                model.model_tree[ec][x][y]["Enzyme"] = ec+"."+x+"."+y
                for z in model.enzymes_tree[ec][x][y]:
                    if model.model_tree[ec][x][y][z]["Tree"] is None:
                        continue
                    model.model_tree[ec][x][y][z]["Enzyme"] = ec+"."+x+"."+y+"."+z
    
    first_level = []
    second_level = []
    third_level = []
    fourth_level = []
    depth = 0
    for ec in model.enzymes_tree:
        if model.model_tree[ec]["Tree"] == None:
            continue
        first_level.append(model.model_tree[ec]["Enzyme"])
        for x in model.enzymes_tree[ec]:
            if model.model_tree[ec][x]["Tree"] == None:
                continue
            second_level.append(model.model_tree[ec][x]["Enzyme"])
            for y in model.enzymes_tree[ec][x]:
                if model.model_tree[ec][x][y]["Tree"] == None:
                    continue
                third_level.append(model.model_tree[ec][x][y]["Enzyme"])
                for z in model.enzymes_tree[ec][x][y]:
                    if model.model_tree[ec][x][y][z]["Tree"] == None:
                        continue
                    fourth_level.append(model.model_tree[ec][x][y][z]["Enzyme"])
    return first_level, second_level, third_level, fourth_level

def get_mean_ap(table, thresh_train=0, thresh_test=0):
    aps = []
    for enzyme in table:
        if table[enzyme]["num_pos_train"] >= thresh_train and table[enzyme]["num_pos_test"] >= thresh_test:
            aps.append(table[enzyme]["AP"])

    return np.nanmean(aps), np.nanstd(aps)

def get_mean_auroc(table, thresh_train=0, thresh_test=0):
    aurocs = []
    for enzyme in table:
        if table[enzyme]["num_pos_train"] >= thresh_train and table[enzyme]["num_pos_test"] >= thresh_test:
            aurocs.append(table[enzyme]["AUROC"])

    return np.nanmean(aurocs), np.nanstd(aurocs)

def get_mean_R_prec(table, thresh_train=0, thresh_test=0):
    R_precs = []
    for enzyme in table:
        if table[enzyme]["num_pos_train"] >= thresh_train and table[enzyme]["num_pos_test"] >= thresh_test:
            R_precs.append(table[enzyme]["R-PREC"])

    return np.nanmean(R_precs), np.nanstd(R_precs)

def test(model, test_dict, est):
    first_level, second_level, third_level, fourth_level = get_list_of_enzymes(model)
    results = {}
    for enzyme in fourth_level:
        if enzyme not in test_dict:
            continue

        results[enzyme] = {}
        ec_breakdown = enzyme.split('.')
        depth = 0
        currpos = model.model_tree
        while depth < len(ec_breakdown):
            if ec_breakdown[depth] not in currpos:
                print("Enzyme not present in the Model at depth %d." % (depth + 1))
                return None
            elif currpos[ec_breakdown[depth]][est] == None:
                print("Enzyme has less than %d Positive examples at depth %d." % (self.min_positive_examples, depth + 1))
                return None
            currpos = currpos[ec_breakdown[depth]]
            depth += 1

        try:
            x_test, y_test, a, b, c, d = test_dict[enzyme]
        except:
            x_test = test_dict[enzyme]["data"]
            y_test = test_dict[enzyme]["target"]
        
        results[enzyme] = {}
        results[enzyme]["num_pos_train"] = currpos["# Positive Examples"]
        results[enzyme]["num_pos_test"] = y_test[y_test == 1.0].shape[0]

        x_test = np.array(test_dict[enzyme]["data"])
        y_true = np.array(test_dict[enzyme]["target"])
        y_hat = model.predict_proba_for_testing(x_test, enzyme, est)

        results[enzyme]['AP'] = sklearn.metrics.average_precision_score(y_true, y_hat)
        results[enzyme]['AUROC'] = sklearn.metrics.roc_auc_score(y_true, y_hat)

        R = y_true[y_true == 1.0].shape[0]
        sorted_indices = np.argsort(y_hat)

        sorted_y_true = y_true[sorted_indices]
        top_R = sorted_y_true[-R:]
        results[enzyme]['R-PREC'] = float(top_R[top_R == 1.0].shape[0])/float(R)
        
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inhibitors', default="False")
    parser.add_argument('--similarity', default="True")
    parser.add_argument('--ratio', default="all")
    parser.add_argument('--test_set', default="Full")
    parser.add_argument('--estimator', default="Tree")
    parser.add_argument('--model_folder', default="./")
    parser.add_argument('--output_template', default="results")
    args = parser.parse_args()

    if args.similarity.lower() == "true":
        sim = "_sim"
    elif args.similarity.lower() == "false":
        sim = ""
    else:
        print("Argument Error: --similarity must be given a valid boolean identifier.")
        exit(1)

    if args.inhibitors.lower() == "true":
        inh = "_inh"
    elif args.inhibitors.lower() == "false":
        inh = ""
    else:
        print("Argument Error: --inhibitors must be given a valid boolean identifier.")
        exit(1)

    r = args.ratio
    est = args.estimator
    output_file = args.output_template + sim + inh + '_r' + r + '.pkl'

    if args.test_set == "Full":
        test_dict = pickle_load("../data/tree_data/tree_and_flat_data%s_test.pkl" % (inh))
    elif args.test_set == "Inhibitor":
        print("Error: TODO add inhibitor test set extraction to this repo.")
    else:
        print("Error: Invalid test set name. Valid names are 'Full' and 'Inhibitor'")
        exit(1)

    model = pickle_load(os.path.join(args.model_folder, "tree_classifier%s%s_r%s.pkl" % (sim, inh, r)))

    results = test(model, test_dict, est)

    pickle_dump(results, output_file)

    mean_ap, std_ap = get_mean_ap(results)
    mean_auroc, std_auroc = get_mean_auroc(results)
    mean_rprec, std_rprec = get_mean_R_prec(results)

    print("AUROC : %.3f +/- %.3f" % (mean_auroc, std_auroc))
    print("AP:     %.3f +/- %.3f" % (mean_ap, std_ap))
    print("R-PREC: %.3f +/- %.3f" % (mean_rprec, std_rprec))

