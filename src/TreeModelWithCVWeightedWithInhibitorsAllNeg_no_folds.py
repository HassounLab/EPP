# import sys
# import requests
import time
import numpy as np
import random
# import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import sklearn
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import GridSearchCV
from GridSearchCV import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
# from testing import calc_confusion_matrix_for_threshold
# from testing import print_perf_metrics_for_threshold
# from testing import print_auc_and_plot_roc
# from testing import make_plot_perf_vs_threshold
# from testing import calc_perf_metrics_for_threshold_weighted
# from testing import calc_perf_metrics_for_threshold
# from rdkit import DataStructs
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit.Chem import MACCSkeys

from sklearn.metrics import make_scorer
import pickle

class EnzymePredictionTreeModelWithCVWeightedWithInhibitorsAllNeg(object):
    '''
    Class for prediction of interaction between molecule and enzyme subclass.
    
    Can choose representation of molecule. Default is maccs.
    '''
    
    def __init__(self,
                   train_filepath,
                   test_filepath,
                   ec=0,
                   rep="maccs",
                   min_positive_examples=10,
                   C=1.0,
                   num_folds=5):
        '''
        Builds instance of dict containing all estimators.
        '''
        self.train_filepath = train_filepath
        self.data_ready = self.pickle_load(train_filepath)
        self.test_data = self.pickle_load(test_filepath)
        print("done loading data")
        self.rep = rep
        self.min_positive_examples = min_positive_examples
        self.num_folds = num_folds
        self.model_tree = {}
        self.C = C
        self.ec = ec
        
    def prepare_data(self):
        self.data_by_ec = {}
        for ec in self.enzymes_tree:
            for x in self.enzymes_tree[ec]:
                for y in self.enzymes_tree[ec][x]:
                    for z in self.enzymes_tree[ec][x][y]:
                        print("Doing EC %s.%s.%s.%s" % (ec, x, y, z), end="\r")
                        xf_NF, yf_N, num_pos_f, sample_weights, inh_test = self.prepare_data(int(ec), x=int(x), y=int(y), z=int(z))
                        if inh_test is None:
                            has_inh = False
                        else:
                            has_inh = True
                        xf_tr_NF_list, yf_tr_N_list, wf_tr_N_list, xf_te_NF_list, yf_te_N_list, wf_te_N_list = self.prepare_folds(xf_NF, yf_N, sample_weights, has_inh)
                        
                        # save this to data_by_ec
                        enzyme = ec+'.'+x+'.'+y+'.'+z
                        self.data_by_ec[enzyme] = [xf_tr_NF_list, yf_tr_N_list, wf_tr_N_list, xf_te_NF_list, yf_te_N_list, wf_te_N_list, inh_test, num_pos_f]
        f = open("data_by_ec_extra_neg.pkl", "wb")
        pickle.dump(self.data_by_ec, f)
        f.close()
    def do_enzymes_match(self, query, reference):
        for i, elem in enumerate(query):
            if elem != reference[i]:
                return False
        return True
    
    def unionize_data(self, query_enzyme):
        new_x_tr_NF_list = [np.array([]), np.array([]), np.array([]), np.array([]), np.array([])]
        new_y_tr_N_list = [np.array([]), np.array([]), np.array([]), np.array([]), np.array([])]
        new_w_tr_N_list = [np.array([]), np.array([]), np.array([]), np.array([]), np.array([])]
        new_x_te_NF_list = [np.array([]), np.array([]), np.array([]), np.array([]), np.array([])]
        new_y_te_N_list = [np.array([]), np.array([]), np.array([]), np.array([]), np.array([])]
        new_w_te_N_list = [np.array([]), np.array([]), np.array([]), np.array([]), np.array([])]
        new_inh_test = None
        num_pos = 0
        query_enzyme_breakdown = query_enzyme.split('.')
        for enzyme in self.data_by_ec:
            enzyme_breakdown = enzyme.split('.')
            
            if self.do_enzymes_match(query_enzyme_breakdown, enzyme_breakdown):
                x_tr_NF_list, y_tr_N_list, w_tr_N_list, x_te_NF_list, y_te_N_list, w_te_N_list, inh_test, num_pos_i = self.data_by_ec[enzyme]
                num_pos += num_pos_i
                
                # inh test does not have folds
                if inh_test is not None:
                    if new_inh_test is None:
                        new_inh_test = np.copy(inh_test)
                    else:
                        new_inh_test = np.vstack((new_inh_test, np.copy(inh_test)))
                
                for i in range(self.num_folds):
                    if new_x_tr_NF_list[i].shape[0] == 0:
                        new_x_tr_NF_list[i] = np.copy(x_tr_NF_list[i])
                    else:
                        new_x_tr_NF_list[i] = np.vstack((new_x_tr_NF_list[i], np.copy(x_tr_NF_list[i])))
                        
                    if new_y_tr_N_list[i].shape[0] == 0:
                        new_y_tr_N_list[i] = np.copy(y_tr_N_list[i])
                    else:
                        new_y_tr_N_list[i] = np.hstack((new_y_tr_N_list[i], np.copy(y_tr_N_list[i])))
                        
                    if new_w_tr_N_list[i].shape[0] == 0:
                        new_w_tr_N_list[i] = np.copy(w_tr_N_list[i])
                    else:
                        new_w_tr_N_list[i] = np.hstack((new_w_tr_N_list[i], np.copy(w_tr_N_list[i])))
                        
                    if new_x_te_NF_list[i].shape[0] == 0:
                        new_x_te_NF_list[i] = np.copy(x_te_NF_list[i])
                    else:
                        new_x_te_NF_list[i] = np.vstack((new_x_te_NF_list[i], np.copy(x_te_NF_list[i])))
                        
                    if new_y_te_N_list[i].shape[0] == 0:
                        new_y_te_N_list[i] = y_te_N_list[i]
                    else:
                        new_y_te_N_list[i] = np.hstack((new_y_te_N_list[i], np.copy(y_te_N_list[i])))
                        
                    if new_w_te_N_list[i].shape[0] == 0:
                        new_w_te_N_list[i] = np.copy(w_te_N_list[i])
                    else:
                        new_w_te_N_list[i] = np.hstack((new_w_te_N_list[i], np.copy(w_te_N_list[i])))
                                            
        # don't remove duplicates for now. Will make runtime longer though. If I remove them, then datasets are likely to be very
        # unlabalanced, thus I need extra steps to rebalance. This is likely to take a long time.
#         for i in range(self.num_folds):
#             new_x_tr_NF_list[i] = self.remove_duplicates_array(new_x_tr_NF_list[i])
        return new_x_tr_NF_list, new_y_tr_N_list, new_w_tr_N_list, new_x_te_NF_list, new_y_te_N_list, new_w_te_N_list, new_inh_test, num_pos
      
    def create_tree(self):        
        null_nodes = 0
        valid_nodes = 0
        for ec in self.enzymes_tree:
            print("Doing EC %s" % ec)
            self.model_tree[ec] = {}
            query_enzyme = ec
            if query_enzyme in self.data_ready and query_enzyme in self.test_data:
                x_NF, y_N, num_pos, sample_weights_bal, sample_weights_sim, inh_test = self.data_ready[query_enzyme]
                x_NF_te, y_N_te, num_pos_te, sample_weights_bal_te, sample_weights_sim_te, inh_test_te = self.test_data[query_enzyme]
            else:
                self.model_tree[ec]["Estimator"] = None
                null_nodes += 1
                continue
            
            self.model_tree[ec]["# Positive Examples"] = num_pos
            
            if num_pos < self.min_positive_examples:
                self.model_tree[ec]["Estimator"] = None
                null_nodes += 1
                continue
            valid_nodes += 1
            
            if inh_test is not None:
                self.model_tree[ec]["Inh Test"] = inh_test 

            self.model_tree[ec]["X Test"] = x_NF_te
            self.model_tree[ec]["Y Test"] = y_N_te
            self.model_tree[ec]["W bal Test"] = sample_weights_bal_te
            self.model_tree[ec]["W sim Test"] = sample_weights_sim_te
            
            first_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=sample_weights_sim, proba=True)
            first_est = first_model_data["best_estimator"]
            self.model_tree[ec]["Estimator"] = first_est
            
            first_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=sample_weights_sim, proba=True)
            first_est = first_model_data["best_estimator"]
            self.model_tree[ec]["Est Ref RFClas"] = first_est
            
            # first_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=sample_weights_bal, proba=True)
            # first_est = first_model_data["best_estimator"]
            # self.model_tree[ec]["Est Ref RFClas No Weight"] = first_est           
        
            # first_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=sample_weights_bal, proba=True)
            # first_est = first_model_data["best_estimator"]
            # self.model_tree[ec]["Est No Weight"] = first_est
            
            for x in self.enzymes_tree[ec]:
                print("\tDoing EC %s.%s" % (ec, x))
                self.model_tree[ec][x] = {}
                query_enzyme = ec+'.'+x
                if query_enzyme in self.data_ready and query_enzyme in self.test_data:
                    x_NF, y_N, num_pos, sample_weights_bal, sample_weights_sim, inh_test = self.data_ready[query_enzyme]
                    x_NF_te, y_N_te, num_pos_te, sample_weights_bal_te, sample_weights_sim_te, inh_test_te = self.test_data[query_enzyme]
                else:
                    self.model_tree[ec][x]["Estimator"] = None
                    null_nodes += 1
                    continue

                self.model_tree[ec][x]["# Positive Examples"] = num_pos

                if num_pos < self.min_positive_examples:
                    self.model_tree[ec][x]["Estimator"] = None
                    null_nodes += 1
                    continue
                valid_nodes += 1

                if inh_test is not None:
                    self.model_tree[ec][x]["Inh Test"] = inh_test 
                
                self.model_tree[ec][x]["X Test"] = x_NF_te
                self.model_tree[ec][x]["Y Test"] = y_N_te
                self.model_tree[ec][x]["W bal Test"] = sample_weights_bal_te
                self.model_tree[ec][x]["W sim Test"] = sample_weights_sim_te
                
                yhat_train_first = self.model_tree[ec]["Estimator"].predict_proba(x_NF)[:,1]
                second_model_data = self.select_model("RFRegr", x_NF, y_N, y_prev=yhat_train_first, sample_weights=sample_weights_sim)
                second_est = second_model_data["best_estimator"]
                self.model_tree[ec][x]["Estimator"] = second_est
                
                second_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=sample_weights_sim, proba=True)
                second_est = second_model_data["best_estimator"]
                self.model_tree[ec][x]["Est Ref RFClas"] = second_est

                # second_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=sample_weights_bal, proba=True)
                # second_est = second_model_data["best_estimator"]
                # self.model_tree[ec][x]["Est Ref RFClas No Weight"] = second_est

                # yhat_train_first = self.model_tree[ec]["Est No Weight"].predict_proba(x_NF)[:,1]
                # second_model_data = self.select_model("RFRegr", x_NF, y_N, y_prev=yhat_train_first, sample_weights=sample_weights_bal)
                # second_est = second_model_data["best_estimator"]
                # self.model_tree[ec][x]["Est No Weight"] = second_est

                for y in self.enzymes_tree[ec][x]:
                    print("\t\tDoing EC %s.%s.%s" % (ec, x, y))
                    self.model_tree[ec][x][y] = {}
                    query_enzyme = ec+'.'+x+'.'+y
                    if query_enzyme in self.data_ready and query_enzyme in self.test_data:
                        x_NF, y_N, num_pos, sample_weights_bal, sample_weights_sim, inh_test = self.data_ready[query_enzyme]
                        x_NF_te, y_N_te, num_pos_te, sample_weights_bal_te, sample_weights_sim_te, inh_test_te = self.test_data[query_enzyme]
                    else:
                        self.model_tree[ec][x][y]["Estimator"] = None
                        null_nodes += 1
                        continue
                    self.model_tree[ec][x][y]["# Positive Examples"] = num_pos
                    
                    if num_pos < self.min_positive_examples:
                        self.model_tree[ec][x][y]["Estimator"] = None
                        null_nodes += 1
                        continue
                    valid_nodes += 1
                    
                    if inh_test is not None:
                        self.model_tree[ec][x][y]["Inh Test"] = inh_test 
                    
                    self.model_tree[ec][x][y]["X Test"] = x_NF_te
                    self.model_tree[ec][x][y]["Y Test"] = y_N_te 
                    self.model_tree[ec][x][y]["W bal Test"] = sample_weights_bal_te
                    self.model_tree[ec][x][y]["W sim Test"] = sample_weights_sim_te
                
                    yhat_train_first = self.model_tree[ec]["Estimator"].predict_proba(x_NF)[:,1]
                    yhat_train_second = self.model_tree[ec][x]["Estimator"].predict(x_NF) + yhat_train_first
                    third_model_data = self.select_model("RFRegr", x_NF, y_N, y_prev=yhat_train_second, sample_weights=sample_weights_sim)
                    third_est = third_model_data["best_estimator"]
                    self.model_tree[ec][x][y]["Estimator"] = third_est

                    third_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=sample_weights_sim, proba=True)
                    third_est = third_model_data["best_estimator"]
                    self.model_tree[ec][x][y]["Est Ref RFClas"] = third_est

                    # third_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=sample_weights_bal, proba=True)
                    # third_est = third_model_data["best_estimator"]
                    # self.model_tree[ec][x][y]["Est Ref RFClas No Weight"] = third_est                      

                    # yhat_train_first = self.model_tree[ec]["Est No Weight"].predict_proba(x_NF)[:,1]
                    # yhat_train_second = self.model_tree[ec][x]["Est No Weight"].predict(x_NF) + yhat_train_first
                    # third_model_data = self.select_model("RFRegr", x_NF, y_N, y_prev=yhat_train_second, sample_weights=sample_weights_bal)
                    # third_est = third_model_data["best_estimator"]
                    # self.model_tree[ec][x][y]["Est No Weight"] = third_est
                    
                    for z in self.enzymes_tree[ec][x][y]:
                        print("\t\t\tDoing EC %s.%s.%s.%s" % (ec, x, y, z))
                        self.model_tree[ec][x][y][z] = {}
                        query_enzyme = ec+'.'+x+'.'+y+'.'+z
                        if query_enzyme in self.data_ready and query_enzyme in self.test_data:
                            x_NF, y_N, num_pos, sample_weights_bal, sample_weights_sim, inh_test = self.data_ready[query_enzyme]
                            x_NF_te, y_N_te, num_pos_te, sample_weights_bal_te, sample_weights_sim_te, inh_test_te = self.test_data[query_enzyme]
                        else:
                            self.model_tree[ec][x][y][z]["Estimator"] = None
                            null_nodes += 1
                            continue

                        self.model_tree[ec][x][y][z]["# Positive Examples"] = num_pos
                        print(num_pos)

                        if num_pos < self.min_positive_examples:
                            self.model_tree[ec][x][y][z]["Estimator"] = None
                            null_nodes += 1
                            continue
                        valid_nodes += 1
                        
                        if inh_test is not None:
                            self.model_tree[ec][x][y][z]["Inh Test"] = inh_test
                        
                        self.model_tree[ec][x][y][z]["X Test"] = x_NF_te
                        self.model_tree[ec][x][y][z]["Y Test"] = y_N_te
                        self.model_tree[ec][x][y][z]["W bal Test"] = sample_weights_bal_te
                        self.model_tree[ec][x][y][z]["W sim Test"] = sample_weights_sim_te

                        yhat_train_first = self.model_tree[ec]["Estimator"].predict_proba(x_NF)[:,1]
                        yhat_train_second = self.model_tree[ec][x]["Estimator"].predict(x_NF) + yhat_train_first
                        yhat_train_third = self.model_tree[ec][x][y]["Estimator"].predict(x_NF) + yhat_train_second
                        fourth_model_data = self.select_model("RFRegr", x_NF, y_N, y_prev=yhat_train_third, sample_weights=sample_weights_sim)
                        fourth_est = fourth_model_data["best_estimator"]
                        self.model_tree[ec][x][y][z]["Estimator"] = fourth_est

                        fourth_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=sample_weights_sim, proba=True)
                        fourth_est = fourth_model_data["best_estimator"]
                        self.model_tree[ec][x][y][z]["Est Ref RFClas"] = fourth_est

                        # fourth_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=sample_weights_bal, proba=True)
                        # fourth_est = fourth_model_data["best_estimator"]
                        # self.model_tree[ec][x][y][z]["Est Ref RFClas No Weight"] = fourth_est

                        # yhat_train_first = self.model_tree[ec]["Est No Weight"].predict_proba(x_NF)[:,1]
                        # yhat_train_second = self.model_tree[ec][x]["Est No Weight"].predict(x_NF) + yhat_train_first
                        # yhat_train_third = self.model_tree[ec][x][y]["Est No Weight"].predict(x_NF) + yhat_train_second
                        # fourth_model_data = self.select_model("RFRegr", x_NF, y_N, y_prev=yhat_train_third, sample_weights=sample_weights_bal)
                        # fourth_est = fourth_model_data["best_estimator"]
                        # self.model_tree[ec][x][y][z]["Est No Weight"] = fourth_est
                            
        print("%d Valid Nodes" % valid_nodes)
        print("%d Null Nodes" % null_nodes)
        
    def prepare_folds(self, x_NF, y_N, sample_weight, has_inh):
        x_pos_NF = x_NF[y_N == 1.0]
        y_pos_N = y_N[y_N == 1.0]
        sample_weight_pos = sample_weight[y_N == 1.0]
        
        x_neg_NF = x_NF[y_N == 0.0]
        y_neg_N = y_N[y_N == 0.0]
        sample_weight_neg = sample_weight[y_N == 0.0]
        
        if has_inh:
            x_inh_NF = x_NF[y_N == 10.0]
            y_inh_N = y_N[y_N == 10.0]
            sample_weight_inh = sample_weight[y_N == 10.0]
        
        x_tr_NF_list_pos, y_tr_N_list_pos, w_tr_N_list_pos, x_va_NF_list_pos, y_va_N_list_pos, w_va_N_list_pos = self.prepare_singleclass_folds(x_pos_NF, y_pos_N, sample_weight_pos, inh=False)
        
        x_tr_NF_list_neg, y_tr_N_list_neg, w_tr_N_list_neg, x_va_NF_list_neg, y_va_N_list_neg, w_va_N_list_neg = self.prepare_singleclass_folds(x_neg_NF, y_neg_N, sample_weight_neg, inh=False)
        
        if has_inh:
            x_tr_NF_list_inh, y_tr_N_list_inh, w_tr_N_list_inh, x_va_NF_list_inh, y_va_N_list_inh, w_va_N_list_inh = self.prepare_singleclass_folds(x_inh_NF, y_inh_N, sample_weight_inh, inh=True)
        
        x_tr_NF_list = []
        y_tr_N_list = []
        w_tr_N_list = []
        x_va_NF_list = []
        y_va_N_list = []
        w_va_N_list = []
        for i in range(self.num_folds):
            if has_inh:
                x_tr_NF_list.append(np.vstack((x_tr_NF_list_pos[i], x_tr_NF_list_neg[i], x_tr_NF_list_inh[i])))
                y_tr_N_list.append(np.hstack((y_tr_N_list_pos[i], y_tr_N_list_neg[i], y_tr_N_list_inh[i])))
                w_tr_N_list.append(np.hstack((w_tr_N_list_pos[i], w_tr_N_list_neg[i], w_tr_N_list_inh[i])))
                x_va_NF_list.append(np.vstack((x_va_NF_list_pos[i], x_va_NF_list_neg[i], x_va_NF_list_inh[i])))
                y_va_N_list.append(np.hstack((y_va_N_list_pos[i], y_va_N_list_neg[i], y_va_N_list_inh[i])))
                w_va_N_list.append(np.hstack((w_va_N_list_pos[i], w_va_N_list_neg[i], w_va_N_list_inh[i])))
            else:
                x_tr_NF_list.append(np.vstack((x_tr_NF_list_pos[i], x_tr_NF_list_neg[i])))
                y_tr_N_list.append(np.hstack((y_tr_N_list_pos[i], y_tr_N_list_neg[i])))
                w_tr_N_list.append(np.hstack((w_tr_N_list_pos[i], w_tr_N_list_neg[i])))
                x_va_NF_list.append(np.vstack((x_va_NF_list_pos[i], x_va_NF_list_neg[i])))
                y_va_N_list.append(np.hstack((y_va_N_list_pos[i], y_va_N_list_neg[i])))
                w_va_N_list.append(np.hstack((w_va_N_list_pos[i], w_va_N_list_neg[i])))
       
        return x_tr_NF_list, y_tr_N_list, w_tr_N_list, x_va_NF_list, y_va_N_list, w_va_N_list
        
    def prepare_singleclass_folds(self, x_NF, y_N, sample_weight, inh=False):
        if inh: # then need to reset labels of inhibitors
            size = y_N.shape[0]
            y_N = np.full(size, 0.0)
        
        N = y_N.size
        n_rows_per_fold = int(np.ceil(N / float(self.num_folds))) * np.ones(self.num_folds, dtype=np.int32)
        n_surplus = np.sum(n_rows_per_fold) - N
        if n_surplus > 0:
            n_rows_per_fold[-n_surplus:] -= 1
        assert np.allclose(np.sum(n_rows_per_fold), N)
        fold_boundaries = np.hstack([0, np.cumsum(n_rows_per_fold)])
        start_per_fold = fold_boundaries[:-1]
        stop_per_fold = fold_boundaries[1:]
        
        x_tr_NF_list = []
        y_tr_N_list = []
        w_tr_N_list = []
        x_va_NF_list = []
        y_va_N_list = []
        w_va_N_list = []
        ## Loop over folds from 1, 2, ... K=num_folds
        for fold_id in range(1, self.num_folds + 1):
            fold_start = start_per_fold[fold_id-1]
            fold_stop = stop_per_fold[fold_id-1]

            # Training data is everything that's not current validation fold
            x_tr_NF = np.vstack([x_NF[:fold_start], x_NF[fold_stop:]])
            y_tr_N = np.hstack([y_N[:fold_start], y_N[fold_stop:]])
            w_tr_N = np.hstack([sample_weight[:fold_start], sample_weight[fold_stop:]])
            
            x_va_NF = x_NF[fold_start:fold_stop].copy()
            y_va_N = y_N[fold_start:fold_stop].copy()
            w_va_N = sample_weight[fold_start:fold_stop].copy()
            
            x_tr_NF_list.append(x_tr_NF)
            y_tr_N_list.append(y_tr_N)
            w_tr_N_list.append(w_tr_N)
            x_va_NF_list.append(x_va_NF)
            y_va_N_list.append(y_va_N)
            w_va_N_list.append(w_va_N)
        
        return x_tr_NF_list, y_tr_N_list, w_tr_N_list, x_va_NF_list, y_va_N_list, w_va_N_list
            
    
    def select_model(self, estimator, x_NF, y_N, y_prev=0, sample_weights=None, proba=False):
        estimator_names = ["LR", "Ridge", "Lasso", "RFClas", "RFRegr", "SVM"]
        if estimator == "LR":
            est = sklearn.linear_model.LogisticRegression(solver='liblinear')
            param_grid = {'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
            proba = True
        elif estimator == "Ridge":
            est = sklearn.linear_model.Ridge()
            param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
            proba = True
        elif estimator == "Lasso":
            est = sklearn.linear_model.Lasso()
            param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
            proba = True
        elif estimator == "RFClas":
            est = sklearn.ensemble.RandomForestClassifier()
            param_grid = {'min_samples_leaf': [1, 5, 10, 20, 50, 100, 200], 'n_estimators': [50], 'max_features': ["sqrt"]}
            proba = True
        elif estimator == "RFRegr":
            est = sklearn.ensemble.RandomForestRegressor()
            param_grid = {'min_samples_leaf': [1, 5, 10, 20, 50, 100, 200], 'n_estimators': [50], 'max_features': ["sqrt"]}
            proba = False
        elif estimator == "SVM":
            est = sklearn.svm.SVC()
            param_grid = {'C': [0.01, 0.1, 1.0, 10.0], 'gamma': [0.01, 0.1, 1.0]}
            proba = False
        else:
            print("Error: Invalid estimator name")
            return

        if proba:
            gs = GridSearchCV(est, param_grid, cv=5, proba=True) #, iid=False, scoring='average_precision')
        else:
            gs = GridSearchCV(est, param_grid, cv=5) #, iid=False, scoring='average_precision')
            
        if type(y_prev) == int:
            if y_prev == 0:
                y_prev = np.full(y_N.size, 0.0)
#                 
#         if depth == 4:
#             gs.fit(x_NF, y_N - y_prev, y_prev, y_N, sample_weights)
#         else:
#             sample_weights = np.full(y_N.size, 1.0)
        gs.fit(x_NF, y_N - y_prev, y_prev, y_N, sample_weights)
#         gs.fit(x_NF, y_N - y_prev, sample_weight=sample_weights)
        
        return dict(
                best_params = gs.best_params_,
                best_score = gs.best_score_,
                best_estimator = gs.best_estimator_,
                proba = proba)
    
    def predict_proba(self, molecules, enzyme):
        # breakdown enzyme into its subclasses
        ec_breakdown = enzyme.split('.')
        depth = 0
        currpos = self.model_tree
        predictions = np.full(self.num_folds, 0.0)
        while depth < len(ec_breakdown):
            if ec_breakdown[depth] not in currpos:
                print("Enzyme not present in the Model at depth %d." % (depth + 1))
                return None
            elif currpos[ec_breakdown[depth]]["Estimator"] == None:
                print("Enzyme has less than %d Positive examples at depth %d." % (self.min_positive_examples, depth + 1))
                return None
            currpos = currpos[ec_breakdown[depth]]
            if depth == 0:
                for i in range(self.num_folds):
                    predictions[i] += currpos["Estimator"][i].predict_proba(molecules)[:,1]
            else:
                for i in range(self.num_folds):
                    predictions[i] += currpos["Estimator"][i].predict(molecules)
            depth += 1
        
        for i in range(self.num_folds):
            predictions[i][predictions[i] < 0.0] = 0.0
            predictions[i][predictions[i] > 1.0] = 1.0
        
        return predictions
    
    def predict_proba_for_testing(self, molecules, enzyme, est="Estimator"):
        # breakdown enzyme into its subclasses                                                                                 
        ec_breakdown = enzyme.split('.')
        depth = 0
        currpos = self.model_tree
        prediction = 0.0
        while depth < len(ec_breakdown):
            if ec_breakdown[depth] not in currpos:
                print("Enzyme not present in the Model at depth %d." % (depth + 1))
                return None
            elif currpos[ec_breakdown[depth]]["Estimator"] == None:
                print("Enzyme has less than %d Positive examples at depth %d." % (self.min_positive_examples, depth + 1))
                return None
            currpos = currpos[ec_breakdown[depth]]
            
            if est == "Estimator" or est == "Est No Weight":
                if depth == 0:
                    prediction += currpos[est].predict_proba(molecules)[:,1]
                else:
                    prediction += currpos[est].predict(molecules)
            elif est == "Est Ref RFClas" or est == "Est Ref RFClas No Weight":
                prediction = currpos[est].predict(molecules)
            depth += 1

        try:
            prediction[prediction < 0.0] = 0.0
            prediction[prediction > 1.0] = 1.0
        except TypeError:
            prediction = float(prediction)
            if prediction < 0.0:
                prediction = 0.0
            elif prediction > 1.0:
                prediction = 1.0
            prediction = np.array([prediction])
            
        return prediction
    
    def perform_similarity_comparison(self, num_pos, ec, x=0, y=0, z=0, average=False):
        '''
        Gets two arrays of MACCS fingerprints to be compared with one another.
        Returns copy of negative data sorted from lowest to highest similarity with x_fixed
            Either sort by average similarity (test the STD maybe) or by MAX similarity
        '''
        pos_data_dict = self.get_data(("smiles_data/EC%d_smiles.txt") % (ec))
        smiles_pos = []
        for enzyme in pos_data_dict:
            ec_breakdown = enzyme.split('.')
            if (x == 0) or (x == int(ec_breakdown[1])):
                if (y == 0) or (y== int(ec_breakdown[2])):
                    if (z == 0) or (z== int(ec_breakdown[3])):
                        smiles_pos += pos_data_dict[enzyme]
        
        smiles_pos = self.remove_duplicates(smiles_pos)
        smiles_unl = np.loadtxt(("negative_smiles/EC%d_negative_smiles.txt") % (ec), dtype=str)
        
        max_score_list = []
        unl_maccs_list = []
        for smile_u in smiles_unl:
            unl_maccs = self.smiles_maccs_table[smile_u]
            if unl_maccs == None:
                continue
            scores = []
            for smile_p in smiles_pos:
                pos_maccs = self.smiles_maccs_table[smile_p]
                if pos_maccs == None:
                    continue
                scores.append(DataStructs.FingerprintSimilarity(pos_maccs, unl_maccs))
                
            max_score_list.append(np.max(scores))
            unl_maccs_list.append(unl_maccs.ToBitString())
        
        unl_maccs_list = self.bit_strings_to_arrays(unl_maccs_list)
        neg_scores = 1 - np.array(max_score_list)
        temp = np.vstack((np.transpose(unl_maccs_list), neg_scores))
        temp = np.transpose(temp)
        
#         index_sorted = np.argsort(max_score_list)
#         temp = temp[index_sorted]
        np.random.shuffle(temp)
        
        unl_maccs_list = temp[:,:-1]
        neg_scores = temp[:,-1]

#         cutoff = int((1.0/np.mean(neg_scores))*num_pos)  # extra neg (balanced biased)
#         cutoff = num_pos  # balanced
#         cutoff = neg_data.shape[0]   # all unlabeled
        
        neg_data = unl_maccs_list[:cutoff]
        neg_weights = (neg_scores[:cutoff])
        pos_for_neg_weights = 1 - neg_scores[:cutoff]
        
        return neg_data, neg_weights, pos_for_neg_weights
        
    def initialize_smiles_maccs_table(self):
        '''
        Keys are smiles, values are maccs
        Uses local files with smiles data
        '''
        self.smiles_maccs_table = {}
        ec1 = np.loadtxt(("negative_smiles/EC%d_negative_smiles.txt") % (1), dtype=str)
        ec2 = np.loadtxt(("negative_smiles/EC%d_negative_smiles.txt") % (2), dtype=str)
        ec3 = np.loadtxt(("negative_smiles/EC%d_negative_smiles.txt") % (3), dtype=str)
        ec4 = np.loadtxt(("negative_smiles/EC%d_negative_smiles.txt") % (4), dtype=str)
        ec5 = np.loadtxt(("negative_smiles/EC%d_negative_smiles.txt") % (5), dtype=str)
        ec6 = np.loadtxt(("negative_smiles/EC%d_negative_smiles.txt") % (6), dtype=str)
        all_ecs = [ec1, ec2, ec3, ec4, ec5, ec6]
        for data in all_ecs:
            for smile in data:
                if smile not in self.smiles_maccs_table:
                    try:
                        self.smiles_maccs_table[smile] = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smile))
                    except:
                        self.smiles_maccs_table[smile] = None
        all_data = []
        for i in range(6):
            ec = i + 1
            all_data.append(self.get_data("smiles_data/EC%d_smiles.txt" % ec))
        # add inhibitors' smiles
        all_data.append(self.get_data("inhibitors/inhibitors_smiles.txt"))
        
        for data in all_data:
            for key in data:
                for smile in data[key]:
                    if smile not in self.smiles_maccs_table:
                        try:
                            self.smiles_maccs_table[smile] = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smile))
                        except:
                            self.smiles_maccs_table[smile] = None                            
            
                            
                            
    # Functions devoted to getting and preparing data
    def prepare_inhibitors(self, x_tr_NF_list, y_tr_N_list, w_tr_N_list, x_te_NF_list, y_te_N_list, w_te_N_list, ec, x=0, y=0, z=0):
        # first get inhibitors
        inhibitors_dict = self.get_data("inhibitors/inhibitors_maccs_no_cofactors.txt")
        inh = []
        for enzyme in inhibitors_dict:
            ec_breakdown = enzyme.split('.')
            if (ec == int(ec_breakdown[0])):
                if (x == 0) or (x == int(ec_breakdown[1])):
                    if (y == 0) or (y == int(ec_breakdown[2])):
                        if (z == 0) or (z == int(ec_breakdown[3])):
                            inh += inhibitors_dict[enzyme]

        inh = self.remove_duplicates(inh)
        num_inh = len(inh)
        
        if num_inh == 0: # if enzyme does not have inhibitors
            return x_tr_NF_list, y_tr_N_list, w_tr_N_list, None        
        
        inh = self.bit_strings_to_arrays(inh)
        
        new_x_tr_NF_list = []
        new_y_tr_N_list = []
        new_w_tr_N_list = []
        new_x_te_NF_list = []
        new_y_te_N_list = []
        new_w_te_N_list = []
        inh_test_list = []
        for i in range(self.num_folds):
            x_tr_NF = x_tr_NF_list[i]
            y_tr_N  = y_tr_N_list[i]
            w_tr_N  = w_tr_N_list[i]
            
            # divide positives and negatives
            x_pos = x_tr_NF[y_tr_N == 1.0]
            x_neg = x_tr_NF[y_tr_N == 0.0]
            y_pos = y_tr_N[y_tr_N == 1.0]
            y_neg = y_tr_N[y_tr_N == 0.0]            
            w_pos = w_tr_N[y_tr_N == 1.0]
            w_neg = w_tr_N[y_tr_N == 0.0]
            num_neg = x_neg.shape[0]
            
            np.random.shuffle(inh)
            
            cutoff = int(0.25 * num_inh) # minimum inhibitors to save for testing
            if cutoff == 0:
                cutoff = 1
            inh_test = inh[:cutoff]
            inh_remaining = inh[cutoff:]
            num_inh_remaining = inh_remaining.shape[0]
            
            cutoff_neg = int(num_neg * 0.33)
            if cutoff_neg == 0:
                cutoff_neg = 1
            
            if num_inh_remaining >= cutoff_neg:
                x_neg[:cutoff_neg] = inh_remaining[:cutoff_neg]
                w_neg[:cutoff_neg] = np.ones(cutoff_neg)
                inh_test = np.vstack((inh_test, inh_remaining[cutoff_neg:]))
            else:
                x_neg[:num_inh_remaining] = inh_remaining
                w_neg[:num_inh_remaining] = np.ones(num_inh_remaining)
            
            # reassemble data
            X = np.vstack((x_pos, x_neg))
            y = np.hstack((y_pos, y_neg))
            w = np.hstack((w_pos, w_neg))
            data = np.vstack((np.transpose(X), y, w))
            data = np.transpose(data)
            np.random.shuffle(data)
            
            x_tr_NF = data[:,:-2]
            y_tr_N = data[:,-2]
            w_tr_N = data[:,-1]
            
            new_x_tr_NF_list.append(x_tr_NF)
            new_y_tr_N_list.append(y_tr_N)
            new_w_tr_N_list.append(w_tr_N)
            inh_test_list.append(inh_test)
            
        return new_x_tr_NF_list, new_y_tr_N_list, new_w_tr_N_list, inh_test_list
    
    def prepare_data(self, ec, x=0, y=0, z=0):
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
        all_data_dict = self.get_data((self.rep+"_data/EC%d_"+self.rep+".txt") % (ec))
        maccs_pos_temp = []
        for enzyme in all_data_dict:
            ec_breakdown = enzyme.split('.')
            if (x == 0) or (x == int(ec_breakdown[1])):
                if (y == 0) or (y== int(ec_breakdown[2])):
                    if (z == 0) or (z== int(ec_breakdown[3])):
                        maccs_pos_temp += all_data_dict[enzyme]

        maccs_pos = self.remove_duplicates(maccs_pos_temp)
        
        # then get inhibitors
        inhibitors_dict = self.get_data("inhibitors/inhibitors_maccs_no_cofactors.txt")
        inh = []
        for enzyme in inhibitors_dict:
            ec_breakdown = enzyme.split('.')
            if (ec == int(ec_breakdown[0])):
                if (x == 0) or (x == int(ec_breakdown[1])):
                    if (y == 0) or (y == int(ec_breakdown[2])):
                        if (z == 0) or (z == int(ec_breakdown[3])):
                            inh += inhibitors_dict[enzyme]

        inh = self.remove_duplicates(inh)
        num_inh = len(inh)
        
        if num_inh == 0:
            num_inh_to_use = 0
            inh_test = None
        else:
            inh = self.bit_strings_to_arrays(inh)

            # save inhibitors for testing
            cutoff = int(0.25 * num_inh) # minimum inhibitors to save for testing
            if cutoff == 0:
                cutoff = 1
            inh_test = inh[:cutoff]
            inh_remaining = inh[cutoff:]
            num_inh_remaining = inh_remaining.shape[0]

            cutoff_neg = int(int(len(maccs_pos)) * 0.33)
            if cutoff_neg == 0:
                cutoff_neg = 1

            if num_inh_remaining >= cutoff_neg:    # if there are more inhibitors than how many we want to have at most
                inh_to_use = inh_remaining[:cutoff_neg]
                inh_test = np.vstack((inh_test, inh_remaining[cutoff_neg:]))
            else:
                inh_to_use = np.copy(inh_remaining)
                
            y_inh = np.full(inh_to_use.shape[0], 10.0) # signal that they are inhibitors for prepare_folds
            inh_weights = np.full(inh_to_use.shape[0], 1.0)
            inh_data = np.vstack((np.transpose(inh_to_use), y_inh, inh_weights))
            inh_data = np.transpose(inh_data)
            num_inh_to_use = int(inh_to_use.shape[0])

        # Then get negative data
#         neg_data_maccs, neg_weights = self.perform_similarity_comparison(len(maccs_pos), ec, x, y, z)
        neg_data_maccs, neg_weights, pos_for_neg_weights = self.perform_similarity_comparison((int(len(maccs_pos))-num_inh_to_use), ec, x, y, z)
    
        extra_N = int(len(maccs_pos))
        
        # Convert bit strings to arrays of binary
        pos_data_maccs = self.bit_strings_to_arrays(maccs_pos)
        pos_weights = np.full(pos_data_maccs.shape[0], 1.0)
        
#         pos_data_maccs = np.vstack((pos_data_maccs, neg_data_maccs)) # [:extra_N]
#         pos_weights = np.hstack((pos_weights, pos_for_neg_weights)) # [:extra_N]

        # Now prepare data to be used for machine learning purposes
        y_pos = np.full(pos_data_maccs.shape[0], 1.0)
        y_neg = np.full(neg_data_maccs.shape[0], 0.0)

        pos_data = np.vstack((np.transpose(pos_data_maccs), y_pos, pos_weights))
        pos_data = np.transpose(pos_data)

        neg_data = np.vstack((np.transpose(neg_data_maccs), y_neg, neg_weights))
        neg_data = np.transpose(neg_data)

        if num_inh == 0:
            all_data = np.vstack((pos_data, neg_data))
        else:
            all_data = np.vstack((pos_data, neg_data, inh_data))
        np.random.shuffle(all_data)

        x_NF = all_data[:,:-2]
        y_N = all_data[:,-2]
        sample_weights = all_data[:,-1]

        return x_NF, y_N, int(pos_data.shape[0]), sample_weights, inh_test #pos_data_maccs, neg_data_maccs, y_pos, y_neg, pos_weights, neg_weights, 
    
    def get_negative_sample_for_EC(self, ec_num, num_samples):
        '''
        Gets sampe of Negative Data based on: representation (eg maccs), ec that is positive, number of samples needed
        '''
        negatives = np.loadtxt(("negative_"+self.rep+"/EC%d_negative_"+self.rep+".txt") % (ec_num), dtype=str)
        sample = []
        indexes = []
        num_total = 0
        while num_total < num_samples:
            index = random.randint(0, num_samples - 1)
            if index in indexes:
                continue
            else:
                sample.append(negatives[index])
                num_total += 1
                indexes.append(index)
        return np.array(sample)
    
    def create_ec_numbers_tree(self):
        '''
        Builds enzymes tree that is used for creating the model
        '''
        
        self.enzymes_tree = {}
        for ec_num in [1, 2, 3, 4, 5, 6]:
            ec_enzymes = list(self.get_data(("utils/EC%d_"+self.rep+".txt") % (ec_num)))
            
            for enzyme in ec_enzymes:
                ec_breakdown = enzyme.split('.')
                ec = ec_breakdown[0]
                x = ec_breakdown[1]
                y = ec_breakdown[2]
                z = ec_breakdown[3]
                
                if ec not in self.enzymes_tree:
                    self.enzymes_tree[ec] = {}
                if x not in self.enzymes_tree[ec]:
                    self.enzymes_tree[ec][x] = {}
                if y not in self.enzymes_tree[ec][x]:
                    self.enzymes_tree[ec][x][y] = {}
                if z not in self.enzymes_tree[ec][x][y]:
                    self.enzymes_tree[ec][x][y][z] = None
                    
    # Functions devoted to Testing
    def print_test_at_node(self, enzyme):
        ec_breakdown = enzyme.split('.')
        depth = 0
        currpos = self.model_tree
        while depth < len(ec_breakdown):
            if ec_breakdown[depth] not in currpos:
                print("Enzyme not present in the Model at depth %d." % (depth + 1))
                return None
            elif currpos[ec_breakdown[depth]]["Estimator"] == None:
                print("Enzyme has less than %d Positive examples at depth %d." % (self.min_positive_examples, depth + 1))
                return None
            currpos = currpos[ec_breakdown[depth]]
            depth += 1
            
        x_test_NF = currpos["X Test"]
        y_test_N = currpos["Y Test"]

        y_hat = self.predict_proba(x_test_NF, enzyme)
        
        print("Testing on %d examples" % x_test_NF.shape[0])
        print()

        try:
            print_perf_metrics_for_threshold(y_test_N, y_hat, 0.5)
        except:
            pass
        try:
            print_auc_and_plot_roc(y_test_N, y_hat)
        except:
            pass
        try:
            make_plot_perf_vs_threshold(y_test_N, y_hat, bin_edges=np.linspace(0, 1, 21))
        except:
            pass
        
        try:
            acc, tpr, tnr, ppv, npv = calc_perf_metrics_for_threshold(y_test_N, y_hat, 0.5)
        except:
            acc, tpr, tnr, ppv, npv = None, None, None, None, None
        try:
            auc = sklearn.metrics.roc_auc_score(y_test_N, y_hat)
        except:
            auc = None
        
        return dict(acc=acc,
                    tpr=tpr,
                    tnr=tnr,
                    ppv=ppv,
                    npv=npv,
                    auc=auc)
    
#     def test_at_node(self, enzyme, est="Estimator"):
#         ec_breakdown = enzyme.split('.')
#         depth = 0
#         currpos = self.model_tree
#         while depth < len(ec_breakdown):
#             if ec_breakdown[depth] not in currpos:
#                 print("Enzyme not present in the Model at depth %d." % (depth + 1))
#                 return None
#             elif currpos[ec_breakdown[depth]]["Estimator"] == None:
#                 print("Enzyme has less than %d Positive examples at depth %d." % (self.min_positive_examples, depth + 1))
#                 return None
#             currpos = currpos[ec_breakdown[depth]]
#             depth += 1
            
#         x_test_NF_list = currpos["X Test"]
#         y_test_N_list = currpos["Y Test"]
#         w_test_N_list = currpos["W Test"]

#         acc_list = []
#         tpr_list = []
#         tnr_list = []
#         ppv_list = []
#         npv_list = []
#         auc_list = []
#         acc_weighted_list = []
#         tpr_weighted_list = []
#         tnr_weighted_list = []
#         ppv_weighted_list = []
#         npv_weighted_list = []
#         auc_weighted_list = []
#         PU_f1_list = []
#         num_testing_list = []
#         tnr_lowest_neg = []
#         tnr_highest_neg = []
#         tnr_lowest_neg_weighted = []
#         tnr_highest_neg_weighted = []
#         predictions = np.array([])
#         weights = np.array([])
#         for i in range(self.num_folds):
#             y_hat = self.predict_proba_for_testing(x_test_NF_list[i], enzyme, i, est)
            
#             predictions = np.concatenate((predictions, y_hat[y_test_N_list[i] == 0.0]))
#             weights = np.concatenate((weights, w_test_N_list[i][y_test_N_list[i] == 0.0]))

#             acc, tpr, tnr, ppv, npv = calc_perf_metrics_for_threshold_weighted(y_test_N_list[i], y_hat, 0.5, sample_weight=w_test_N_list[i])
#             acc_weighted_list.append(acc)
#             tpr_weighted_list.append(tpr)
#             tnr_weighted_list.append(tnr)
#             ppv_weighted_list.append(ppv)
#             npv_weighted_list.append(npv)
            
#             acc, tpr, tnr, ppv, npv = calc_perf_metrics_for_threshold(y_test_N_list[i], y_hat, 0.5)
#             acc_list.append(acc)
#             tpr_list.append(tpr)
#             tnr_list.append(tnr)
#             ppv_list.append(ppv)
#             npv_list.append(npv)
            
#             neg_pred = y_hat[y_test_N_list[i] == 0.0]
#             neg_true = y_test_N_list[i][y_test_N_list[i] == 0.0]
#             neg_weights = w_test_N_list[i][y_test_N_list[i] == 0.0]
            
#             index_weights = np.argsort(neg_weights)   # Sorts from lowest to highest
#             neg_pred_sorted = neg_pred[index_weights]
#             neg_true_sorted = neg_true[index_weights]
#             neg_weights_sorted = neg_weights[index_weights]
            
#             low_cutoff = int(0.25*neg_pred.size)
#             high_cutoff = neg_pred.size - low_cutoff
            
#             if low_cutoff == 0:
#                 low_cutoff = 1
#                 high_cutoff -= 1
                
#             acc, tpr, tnr, ppv, npv = calc_perf_metrics_for_threshold(neg_true_sorted[:low_cutoff], neg_pred_sorted[:low_cutoff], 0.5)
#             tnr_lowest_neg.append(tnr)
            
#             acc, tpr, tnr, ppv, npv = calc_perf_metrics_for_threshold(neg_true_sorted[high_cutoff:], neg_pred_sorted[high_cutoff:], 0.5)
#             tnr_highest_neg.append(tnr)
            
#             acc, tpr, tnr, ppv, npv = calc_perf_metrics_for_threshold_weighted(neg_true_sorted[:low_cutoff], neg_pred_sorted[:low_cutoff], 0.5, sample_weight=neg_weights_sorted[:low_cutoff])
#             tnr_lowest_neg_weighted.append(tnr)
            
#             acc, tpr, tnr, ppv, npv = calc_perf_metrics_for_threshold_weighted(neg_true_sorted[high_cutoff:], neg_pred_sorted[high_cutoff:], 0.5, sample_weight=neg_weights_sorted[high_cutoff:])
#             tnr_highest_neg_weighted.append(tnr)

#             try:
#                 auc = sklearn.metrics.roc_auc_score(y_test_N_list[i], y_hat, sample_weight=w_test_N_list[i])
#                 auc_weighted_list.append(auc)
#             except ValueError:
#                 pass
            
#             try:
#                 auc = sklearn.metrics.roc_auc_score(y_test_N_list[i], y_hat)
#                 auc_list.append(auc)
#             except ValueError:
#                 pass
            
#             try:
#                 PU_f1_list.append(PU_F1_approx_for_model_selection(y_hat, y_test_N_list[i]))
#             except:
#                 pass
            
#             num_testing_list.append(currpos["X Test"][i].shape[0])
        
#         return acc_list, tpr_list, tnr_list, ppv_list, npv_list, auc_list, acc_weighted_list, tpr_weighted_list, tnr_weighted_list, ppv_weighted_list, npv_weighted_list, auc_weighted_list, PU_f1_list, num_testing_list, tnr_lowest_neg, tnr_highest_neg, tnr_lowest_neg_weighted, tnr_highest_neg_weighted

    def test_at_node(self, enzyme, est="Estimator", weighted=False):
        ec_breakdown = enzyme.split('.')
        depth = 0
        currpos = self.model_tree
        while depth < len(ec_breakdown):
            if ec_breakdown[depth] not in currpos:
                print("Enzyme not present in the Model at depth %d." % (depth + 1))
                return None
            elif currpos[ec_breakdown[depth]]["Estimator"] == None:
                print("Enzyme has less than %d Positive examples at depth %d." % (self.min_positive_examples, depth + 1))
                return None
            currpos = currpos[ec_breakdown[depth]]
            depth += 1
        x_test_NF_list = currpos["X Test"]
        y_test_N_list = currpos["Y Test"]
        w_test_N_list = currpos["W Test"]
        
        confusion_matrices = []
        auc_list = []
        bottom_25_neg_cm = []
        top_25_neg_cm = []
        predictions = np.array([])
        weights = np.array([])
        for i in range(self.num_folds):
            y_hat = self.predict_proba_for_testing(x_test_NF_list[i], enzyme, i, est)
            
            predictions = np.concatenate((predictions, y_hat[y_test_N_list[i] == 0.0]))
            weights = np.concatenate((weights, w_test_N_list[i][y_test_N_list[i] == 0.0]))
        
            neg_pred = y_hat[y_test_N_list[i] == 0.0]
            neg_true = y_test_N_list[i][y_test_N_list[i] == 0.0]
            neg_weights = w_test_N_list[i][y_test_N_list[i] == 0.0]
            
            index_weights = np.argsort(neg_weights)   # Sorts from lowest to highest
            neg_pred_sorted = neg_pred[index_weights]
            neg_true_sorted = neg_true[index_weights]
            neg_weights_sorted = neg_weights[index_weights]
            
            low_cutoff = int(0.25*neg_pred.size)
            high_cutoff = neg_pred.size - low_cutoff
            
            if low_cutoff == 0:
                low_cutoff = 1
                high_cutoff -= 1
            
            if weighted:
                confusion_matrices.append(sklearn.metrics.confusion_matrix(y_test_N_list[i], y_hat >= 0.5, labels=[0.0, 1.0], sample_weight=w_test_N_list[i]))
                auc_list.append(sklearn.metrics.roc_auc_score(y_test_N_list[i], y_hat, sample_weight=w_test_N_list[i]))
                bottom_25_neg_cm.append(sklearn.metrics.confusion_matrix(neg_true_sorted[:low_cutoff], neg_pred_sorted[:low_cutoff] >= 0.5, labels=[0.0, 1.0], sample_weight=neg_weights_sorted[:low_cutoff]))
                top_25_neg_cm.append(sklearn.metrics.confusion_matrix(neg_true_sorted[high_cutoff:], neg_pred_sorted[high_cutoff:] >= 0.5, labels=[0.0, 1.0], sample_weight=neg_weights_sorted[high_cutoff:]))
            else:
                confusion_matrices.append(sklearn.metrics.confusion_matrix(y_test_N_list[i], y_hat >= 0.5, labels=[0.0, 1.0]))
                auc_list.append(sklearn.metrics.roc_auc_score(y_test_N_list[i], y_hat))
                bottom_25_neg_cm.append(sklearn.metrics.confusion_matrix(neg_true_sorted[:low_cutoff], neg_pred_sorted[:low_cutoff] >= 0.5, labels=[0.0, 1.0]))
                top_25_neg_cm.append(sklearn.metrics.confusion_matrix(neg_true_sorted[high_cutoff:], neg_pred_sorted[high_cutoff:] >= 0.5, labels=[0.0, 1.0]))
                
        return confusion_matrices, auc_list, bottom_25_neg_cm, top_25_neg_cm, predictions, weights
            
    
    def test_all_tree(self, est, weighted):
#         metrics = ["ACC", "TPR", "TNR", "PPV", "NPV", "AUC", "ACC Weighted", "TPR Weighted", "TNR Weighted", "PPV Weighted", "NPV Weighted", "AUC Weighted", "PU_F1", "Num Testing", "TNR Lowest 25%", "TNR Highest 25%", "TNR Lowest 25% Weighted", "TNR Highest 25% Weighted"]
        metrics = ["Confusion Matrices", "AUC List", "Lowest 25% Confidence CM", "Highest 25% Confidence CM", "Neg Predictions", "Neg Weights"]
        for ec in self.enzymes_tree:
            if self.model_tree[ec]["Estimator"] == None:
                continue
            
            enzyme = ec
            results = self.test_at_node(enzyme, est, weighted)
            self.model_tree[ec]["Enzyme"] = enzyme
            for m, metric in enumerate(metrics):
                self.model_tree[ec][est + " " + metric] = results[m]
            for x in self.enzymes_tree[ec]:
                if self.model_tree[ec][x]["Estimator"] == None:
                    continue

                enzyme = ec+'.'+x
                results = self.test_at_node(enzyme, est, weighted)
                self.model_tree[ec][x]["Enzyme"] = enzyme
                for m, metric in enumerate(metrics):
                    self.model_tree[ec][x][est + " " + metric] = results[m]
                for y in self.enzymes_tree[ec][x]:
                    if self.model_tree[ec][x][y]["Estimator"] == None:
                        continue

                    enzyme = ec+'.'+x+'.'+y
                    results = self.test_at_node(enzyme, est, weighted)
                    self.model_tree[ec][x][y]["Enzyme"] = enzyme
                    for m, metric in enumerate(metrics):
                        self.model_tree[ec][x][y][est + " " + metric] = results[m]
                    for z in self.enzymes_tree[ec][x][y]:
                        if self.model_tree[ec][x][y][z]["Estimator"] == None:
                            continue

                        enzyme = ec+'.'+x+'.'+y+'.'+z
                        results = self.test_at_node(enzyme, est, weighted)
                        self.model_tree[ec][x][y][z]["Enzyme"] = enzyme
                        for m, metric in enumerate(metrics):
                            self.model_tree[ec][x][y][z][est + " " + metric] = results[m]
    
    # Utilities
    def get_data(self, filepath):
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
        
    def concatenate(self, alist, divisor):
        '''
        Util for write_data
        '''
        string = ""
        count = 0
        for elem in alist:
            if count != 0:
                string += divisor
            string += elem
            count += 1
        return string
    
    def bit_strings_to_arrays(self, bit_strings):
        bit_arrays = []
        for bit_string in bit_strings:
            bit_arrays.append(np.array([int(i) for i in bit_string]))
        return np.array(bit_arrays)
    
    def remove_duplicates(self, alist):
        newlist = []
        for elem in alist:
            if elem not in newlist:
                newlist.append(elem)
        return newlist
    
    def pickle_load(self, filepath):
        f = open(filepath, "rb")
        data = pickle.load(f)
        f.close()
        return data

    def pickle_dump(self, data, filepath):
        f = open(filepath, "rb")
        pickle.dump(data, f)
        f.close()
    
def PU_F1_approx_for_model_selection(yhat, ytrue):
    '''
    Computes approximation of F1 score.
    Note that this score can be greater than 1.0.
        However, since positive and unlabelled are balanced, the optimal score is at 2.0
    '''    
    ytrue_1 = float(ytrue[ytrue == 1.0].size)
    TP = 0.0
    for i in range(ytrue.size):
        if yhat[i] == 1.0 and ytrue[i] == 1.0:
            TP += 1.0
    try:
        recall = TP / ytrue_1
    except:
        recall = 0.0

    yhat_1_ratio = float(yhat[yhat == 1.0].size) / float(yhat.size)

    try:
        result = (recall * recall) / yhat_1_ratio
    except:
        result = 0.0
    
    return result

        
        
    
    
    

    
    
    
    
    
    
    
    