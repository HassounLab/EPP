
import numpy as np
import sklearn
from GridSearchCV import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from utils import pickle_load, pickle_dump, get_data

class TreeAndFlatModels(object):
    '''
    Class containing Tree and Flat estimators to predict interaction between molecules
    and Enzymes following EC classification.
    '''
    
    def __init__(self,
                   train_filepath,
                   test_filepath,
                   rep="maccs",
                   min_positive_examples=10):
        '''
        Builds instance of dict containing all estimators.
        '''
        self.train_filepath = train_filepath
        self.test_filepath = test_filepath
        self.data_ready = pickle_load(train_filepath)
        self.test_data = pickle_load(test_filepath)
        print("done loading data")
        self.rep = rep
        self.min_positive_examples = min_positive_examples
        self.num_folds = num_folds
        self.model_tree = {}

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
                self.model_tree[ec]["Tree"] = None
                null_nodes += 1
                continue
            
            self.model_tree[ec]["# Positive Examples"] = num_pos
            
            if num_pos < self.min_positive_examples:
                self.model_tree[ec]["Tree"] = None
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
            self.model_tree[ec]["Tree"] = first_est
            
            first_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=sample_weights_sim, proba=True)
            first_est = first_model_data["best_estimator"]
            self.model_tree[ec]["Flat"] = first_est
            
            # first_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=sample_weights_bal, proba=True)
            # first_est = first_model_data["best_estimator"]
            # self.model_tree[ec]["Flat No Weight"] = first_est           
        
            # first_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=sample_weights_bal, proba=True)
            # first_est = first_model_data["best_estimator"]
            # self.model_tree[ec]["Tree No Weight"] = first_est
            
            for x in self.enzymes_tree[ec]:
                print("\tDoing EC %s.%s" % (ec, x))
                self.model_tree[ec][x] = {}
                query_enzyme = ec+'.'+x
                if query_enzyme in self.data_ready and query_enzyme in self.test_data:
                    x_NF, y_N, num_pos, sample_weights_bal, sample_weights_sim, inh_test = self.data_ready[query_enzyme]
                    x_NF_te, y_N_te, num_pos_te, sample_weights_bal_te, sample_weights_sim_te, inh_test_te = self.test_data[query_enzyme]
                else:
                    self.model_tree[ec][x]["Tree"] = None
                    null_nodes += 1
                    continue

                self.model_tree[ec][x]["# Positive Examples"] = num_pos

                if num_pos < self.min_positive_examples:
                    self.model_tree[ec][x]["Tree"] = None
                    null_nodes += 1
                    continue
                valid_nodes += 1

                if inh_test is not None:
                    self.model_tree[ec][x]["Inh Test"] = inh_test 
                
                self.model_tree[ec][x]["X Test"] = x_NF_te
                self.model_tree[ec][x]["Y Test"] = y_N_te
                self.model_tree[ec][x]["W bal Test"] = sample_weights_bal_te
                self.model_tree[ec][x]["W sim Test"] = sample_weights_sim_te
                
                yhat_train_first = self.model_tree[ec]["Tree"].predict_proba(x_NF)[:,1]
                second_model_data = self.select_model("RFRegr", x_NF, y_N, y_prev=yhat_train_first, sample_weights=sample_weights_sim)
                second_est = second_model_data["best_estimator"]
                self.model_tree[ec][x]["Tree"] = second_est
                
                second_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=sample_weights_sim, proba=True)
                second_est = second_model_data["best_estimator"]
                self.model_tree[ec][x]["Flat"] = second_est

                # second_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=sample_weights_bal, proba=True)
                # second_est = second_model_data["best_estimator"]
                # self.model_tree[ec][x]["Flat No Weight"] = second_est

                # yhat_train_first = self.model_tree[ec]["Tree No Weight"].predict_proba(x_NF)[:,1]
                # second_model_data = self.select_model("RFRegr", x_NF, y_N, y_prev=yhat_train_first, sample_weights=sample_weights_bal)
                # second_est = second_model_data["best_estimator"]
                # self.model_tree[ec][x]["Tree No Weight"] = second_est

                for y in self.enzymes_tree[ec][x]:
                    print("\t\tDoing EC %s.%s.%s" % (ec, x, y))
                    self.model_tree[ec][x][y] = {}
                    query_enzyme = ec+'.'+x+'.'+y
                    if query_enzyme in self.data_ready and query_enzyme in self.test_data:
                        x_NF, y_N, num_pos, sample_weights_bal, sample_weights_sim, inh_test = self.data_ready[query_enzyme]
                        x_NF_te, y_N_te, num_pos_te, sample_weights_bal_te, sample_weights_sim_te, inh_test_te = self.test_data[query_enzyme]
                    else:
                        self.model_tree[ec][x][y]["Tree"] = None
                        null_nodes += 1
                        continue
                    self.model_tree[ec][x][y]["# Positive Examples"] = num_pos
                    
                    if num_pos < self.min_positive_examples:
                        self.model_tree[ec][x][y]["Tree"] = None
                        null_nodes += 1
                        continue
                    valid_nodes += 1
                    
                    if inh_test is not None:
                        self.model_tree[ec][x][y]["Inh Test"] = inh_test 
                    
                    self.model_tree[ec][x][y]["X Test"] = x_NF_te
                    self.model_tree[ec][x][y]["Y Test"] = y_N_te 
                    self.model_tree[ec][x][y]["W bal Test"] = sample_weights_bal_te
                    self.model_tree[ec][x][y]["W sim Test"] = sample_weights_sim_te
                
                    yhat_train_first = self.model_tree[ec]["Tree"].predict_proba(x_NF)[:,1]
                    yhat_train_second = self.model_tree[ec][x]["Tree"].predict(x_NF) + yhat_train_first
                    third_model_data = self.select_model("RFRegr", x_NF, y_N, y_prev=yhat_train_second, sample_weights=sample_weights_sim)
                    third_est = third_model_data["best_estimator"]
                    self.model_tree[ec][x][y]["Tree"] = third_est

                    third_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=sample_weights_sim, proba=True)
                    third_est = third_model_data["best_estimator"]
                    self.model_tree[ec][x][y]["Flat"] = third_est

                    # third_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=sample_weights_bal, proba=True)
                    # third_est = third_model_data["best_estimator"]
                    # self.model_tree[ec][x][y]["Flat No Weight"] = third_est                      

                    # yhat_train_first = self.model_tree[ec]["Tree No Weight"].predict_proba(x_NF)[:,1]
                    # yhat_train_second = self.model_tree[ec][x]["Tree No Weight"].predict(x_NF) + yhat_train_first
                    # third_model_data = self.select_model("RFRegr", x_NF, y_N, y_prev=yhat_train_second, sample_weights=sample_weights_bal)
                    # third_est = third_model_data["best_estimator"]
                    # self.model_tree[ec][x][y]["Tree No Weight"] = third_est
                    
                    for z in self.enzymes_tree[ec][x][y]:
                        print("\t\t\tDoing EC %s.%s.%s.%s" % (ec, x, y, z))
                        self.model_tree[ec][x][y][z] = {}
                        query_enzyme = ec+'.'+x+'.'+y+'.'+z
                        if query_enzyme in self.data_ready and query_enzyme in self.test_data:
                            x_NF, y_N, num_pos, sample_weights_bal, sample_weights_sim, inh_test = self.data_ready[query_enzyme]
                            x_NF_te, y_N_te, num_pos_te, sample_weights_bal_te, sample_weights_sim_te, inh_test_te = self.test_data[query_enzyme]
                        else:
                            self.model_tree[ec][x][y][z]["Tree"] = None
                            null_nodes += 1
                            continue

                        self.model_tree[ec][x][y][z]["# Positive Examples"] = num_pos
                        print(num_pos)

                        if num_pos < self.min_positive_examples:
                            self.model_tree[ec][x][y][z]["Tree"] = None
                            null_nodes += 1
                            continue
                        valid_nodes += 1
                        
                        if inh_test is not None:
                            self.model_tree[ec][x][y][z]["Inh Test"] = inh_test
                        
                        self.model_tree[ec][x][y][z]["X Test"] = x_NF_te
                        self.model_tree[ec][x][y][z]["Y Test"] = y_N_te
                        self.model_tree[ec][x][y][z]["W bal Test"] = sample_weights_bal_te
                        self.model_tree[ec][x][y][z]["W sim Test"] = sample_weights_sim_te

                        yhat_train_first = self.model_tree[ec]["Tree"].predict_proba(x_NF)[:,1]
                        yhat_train_second = self.model_tree[ec][x]["Tree"].predict(x_NF) + yhat_train_first
                        yhat_train_third = self.model_tree[ec][x][y]["Tree"].predict(x_NF) + yhat_train_second
                        fourth_model_data = self.select_model("RFRegr", x_NF, y_N, y_prev=yhat_train_third, sample_weights=sample_weights_sim)
                        fourth_est = fourth_model_data["best_estimator"]
                        self.model_tree[ec][x][y][z]["Tree"] = fourth_est

                        fourth_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=sample_weights_sim, proba=True)
                        fourth_est = fourth_model_data["best_estimator"]
                        self.model_tree[ec][x][y][z]["Flat"] = fourth_est

                        # fourth_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=sample_weights_bal, proba=True)
                        # fourth_est = fourth_model_data["best_estimator"]
                        # self.model_tree[ec][x][y][z]["Flat No Weight"] = fourth_est

                        # yhat_train_first = self.model_tree[ec]["Tree No Weight"].predict_proba(x_NF)[:,1]
                        # yhat_train_second = self.model_tree[ec][x]["Tree No Weight"].predict(x_NF) + yhat_train_first
                        # yhat_train_third = self.model_tree[ec][x][y]["Tree No Weight"].predict(x_NF) + yhat_train_second
                        # fourth_model_data = self.select_model("RFRegr", x_NF, y_N, y_prev=yhat_train_third, sample_weights=sample_weights_bal)
                        # fourth_est = fourth_model_data["best_estimator"]
                        # self.model_tree[ec][x][y][z]["Tree No Weight"] = fourth_est
                            
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
            
    
    def select_model(self, estimator, x_NF, y_N, y_prev=0, sample_weights=None, proba=False):
        estimator_names = ["LR", "Ridge", "Lasso", "RFClas", "RFRegr", "SVM"]
        elif estimator == "RFClas":
            est = sklearn.ensemble.RandomForestClassifier()
            param_grid = {'min_samples_leaf': [1, 5, 10, 20, 50, 100, 200], 'n_estimators': [50], 'max_features': ["sqrt"]}
            proba = True
        elif estimator == "RFRegr":
            est = sklearn.ensemble.RandomForestRegressor()
            param_grid = {'min_samples_leaf': [1, 5, 10, 20, 50, 100, 200], 'n_estimators': [50], 'max_features': ["sqrt"]}
            proba = False
        else:
            print("Error: Invalid estimator name")
            return

        if proba:
            gs = GridSearchCV(est, param_grid, cv=5, proba=True)
        else:
            gs = GridSearchCV(est, param_grid, cv=5)
            
        if type(y_prev) == int:
            if y_prev == 0:
                y_prev = np.full(y_N.size, 0.0)

        gs.fit(x_NF, y_N - y_prev, y_prev, y_N, sample_weights)
        
        return dict(
                best_params = gs.best_params_,
                best_score = gs.best_score_,
                best_estimator = gs.best_estimator_,
                proba = proba)
    
    def predict_proba(self, molecules, enzyme, est="Tree"):
        # breakdown enzyme into its subclasses                                                                                 
        ec_breakdown = enzyme.split('.')
        depth = 0
        currpos = self.model_tree
        prediction = 0.0
        while depth < len(ec_breakdown):
            if ec_breakdown[depth] not in currpos:
                print("Enzyme not present in the Model at depth %d." % (depth + 1))
                return None
            elif currpos[ec_breakdown[depth]]["Tree"] == None:
                print("Enzyme has less than %d Positive examples at depth %d." % (self.min_positive_examples, depth + 1))
                return None
            currpos = currpos[ec_breakdown[depth]]
            
            if est == "Tree" or est == "Tree No Weight":
                if depth == 0:
                    prediction += currpos[est].predict_proba(molecules)[:,1]
                else:
                    prediction += currpos[est].predict(molecules)
            elif est == "Flat" or est == "Flat No Weight":
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
    
    def create_ec_numbers_tree(self):
        '''
        Builds enzymes tree that is used for creating the model
        '''
        
        self.enzymes_tree = {}
        for ec_num in [1, 2, 3, 4, 5, 6]:
            ec_enzymes = list(get_data(("utils/EC%d_"+self.rep+".txt") % (ec_num)))
            
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
    
    # Utilities        
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
