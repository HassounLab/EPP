
import numpy as np
import sklearn
from GridSearchCV import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from utils import pickle_load, pickle_dump, bit_strings_to_arrays, remove_duplicates

class TreeAndFlatModels(object):
    '''
    Class containing Tree and Flat estimators to predict interaction between molecules
    and Enzymes following EC classification.
    '''
    
    def __init__(self,
                   train_filepath,
                   test_filepath,
                   rep="maccs",
                   similarity=True,
                   min_positive_examples=10):
        self.train_filepath = train_filepath
        self.test_filepath = test_filepath
        self.data_ready = pickle_load(train_filepath)
        self.test_data = pickle_load(test_filepath)
        print("Done loading data")
        self.similarity = similarity
        self.rep = rep
        self.min_positive_examples = min_positive_examples
        self.model_tree = {}

    def create_tree(self):        
        for ec in self.enzymes_tree:
            print("Doing EC %s" % ec)
            self.model_tree[ec] = {}
            query_enzyme = ec
            if query_enzyme in self.data_ready and query_enzyme in self.test_data:
                x_NF, y_N, num_pos, sample_weights_bal, sample_weights_sim = self.data_ready[query_enzyme]
                x_NF_te, y_N_te, num_pos_te, sample_weights_bal_te, sample_weights_sim_te = self.test_data[query_enzyme]
            else:
                self.model_tree[ec]["Tree"] = None
                continue

            if self.similarity:
                weights = np.copy(sample_weights_sim)
            else:
                weights = np.copy(sample_weights_bal)
            
            self.model_tree[ec]["# Positive Examples"] = num_pos
            self.model_tree[ec]["X Test"] = x_NF_te
            self.model_tree[ec]["Y Test"] = y_N_te
            self.model_tree[ec]["W bal Test"] = sample_weights_bal_te
            self.model_tree[ec]["W sim Test"] = sample_weights_sim_te
            
            first_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=weights, proba=True)
            first_est = first_model_data["best_estimator"]
            self.model_tree[ec]["Tree"] = first_est
            
            first_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=weights, proba=True)
            first_est = first_model_data["best_estimator"]
            self.model_tree[ec]["Flat"] = first_est
            
            for x in self.enzymes_tree[ec]:
                print("\tDoing EC %s.%s" % (ec, x))
                self.model_tree[ec][x] = {}
                query_enzyme = ec+'.'+x
                if query_enzyme in self.data_ready and query_enzyme in self.test_data:
                    x_NF, y_N, num_pos, sample_weights_bal, sample_weights_sim = self.data_ready[query_enzyme]
                    x_NF_te, y_N_te, num_pos_te, sample_weights_bal_te, sample_weights_sim_te = self.test_data[query_enzyme]
                else:
                    self.model_tree[ec][x]["Tree"] = None
                    continue

                if self.similarity:
                    weights = np.copy(sample_weights_sim)
                else:
                    weights = np.copy(sample_weights_bal)

                self.model_tree[ec][x]["# Positive Examples"] = num_pos
                
                self.model_tree[ec][x]["X Test"] = x_NF_te
                self.model_tree[ec][x]["Y Test"] = y_N_te
                self.model_tree[ec][x]["W bal Test"] = sample_weights_bal_te
                self.model_tree[ec][x]["W sim Test"] = sample_weights_sim_te
                
                yhat_train_first = self.model_tree[ec]["Tree"].predict_proba(x_NF)[:,1]
                second_model_data = self.select_model("RFRegr", x_NF, y_N, y_prev=yhat_train_first, sample_weights=weights)
                second_est = second_model_data["best_estimator"]
                self.model_tree[ec][x]["Tree"] = second_est
                
                second_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=weights, proba=True)
                second_est = second_model_data["best_estimator"]
                self.model_tree[ec][x]["Flat"] = second_est

                for y in self.enzymes_tree[ec][x]:
                    print("\t\tDoing EC %s.%s.%s" % (ec, x, y))
                    self.model_tree[ec][x][y] = {}
                    query_enzyme = ec+'.'+x+'.'+y
                    if query_enzyme in self.data_ready and query_enzyme in self.test_data:
                        x_NF, y_N, num_pos, sample_weights_bal, sample_weights_sim = self.data_ready[query_enzyme]
                        x_NF_te, y_N_te, num_pos_te, sample_weights_bal_te, sample_weights_sim_te = self.test_data[query_enzyme]
                    else:
                        self.model_tree[ec][x][y]["Tree"] = None
                        continue

                    if self.similarity:
                        weights = np.copy(sample_weights_sim)
                    else:
                        weights = np.copy(sample_weights_bal)

                    self.model_tree[ec][x][y]["# Positive Examples"] = num_pos
                    
                    self.model_tree[ec][x][y]["X Test"] = x_NF_te
                    self.model_tree[ec][x][y]["Y Test"] = y_N_te 
                    self.model_tree[ec][x][y]["W bal Test"] = sample_weights_bal_te
                    self.model_tree[ec][x][y]["W sim Test"] = sample_weights_sim_te
                
                    yhat_train_first = self.model_tree[ec]["Tree"].predict_proba(x_NF)[:,1]
                    yhat_train_second = self.model_tree[ec][x]["Tree"].predict(x_NF) + yhat_train_first
                    third_model_data = self.select_model("RFRegr", x_NF, y_N, y_prev=yhat_train_second, sample_weights=weights)
                    third_est = third_model_data["best_estimator"]
                    self.model_tree[ec][x][y]["Tree"] = third_est

                    third_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=weights, proba=True)
                    third_est = third_model_data["best_estimator"]
                    self.model_tree[ec][x][y]["Flat"] = third_est
                    
                    for z in self.enzymes_tree[ec][x][y]:
                        print("\t\t\tDoing EC %s.%s.%s.%s" % (ec, x, y, z))
                        self.model_tree[ec][x][y][z] = {}
                        query_enzyme = ec+'.'+x+'.'+y+'.'+z
                        if query_enzyme in self.data_ready and query_enzyme in self.test_data:
                            x_NF, y_N, num_pos, sample_weights_bal, sample_weights_sim = self.data_ready[query_enzyme]
                            x_NF_te, y_N_te, num_pos_te, sample_weights_bal_te, sample_weights_sim_te = self.test_data[query_enzyme]
                        else:
                            self.model_tree[ec][x][y][z]["Tree"] = None
                            continue

                        if self.similarity:
                            weights = np.copy(sample_weights_sim)
                        else:
                            weights = np.copy(sample_weights_bal)

                        self.model_tree[ec][x][y][z]["# Positive Examples"] = num_pos
                        
                        self.model_tree[ec][x][y][z]["X Test"] = x_NF_te
                        self.model_tree[ec][x][y][z]["Y Test"] = y_N_te
                        self.model_tree[ec][x][y][z]["W bal Test"] = sample_weights_bal_te
                        self.model_tree[ec][x][y][z]["W sim Test"] = sample_weights_sim_te

                        yhat_train_first = self.model_tree[ec]["Tree"].predict_proba(x_NF)[:,1]
                        yhat_train_second = self.model_tree[ec][x]["Tree"].predict(x_NF) + yhat_train_first
                        yhat_train_third = self.model_tree[ec][x][y]["Tree"].predict(x_NF) + yhat_train_second
                        fourth_model_data = self.select_model("RFRegr", x_NF, y_N, y_prev=yhat_train_third, sample_weights=weights)
                        fourth_est = fourth_model_data["best_estimator"]
                        self.model_tree[ec][x][y][z]["Tree"] = fourth_est

                        fourth_model_data = self.select_model("RFClas", x_NF, y_N, sample_weights=weights, proba=True)
                        fourth_est = fourth_model_data["best_estimator"]
                        self.model_tree[ec][x][y][z]["Flat"] = fourth_est

    
    def select_model(self, estimator, x_NF, y_N, y_prev=0, sample_weights=None, proba=False):
        estimator_names = ["LR", "Ridge", "Lasso", "RFClas", "RFRegr", "SVM"]
        if estimator == "RFClas":
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
            
            if est == "Tree":
                if depth == 0:
                    prediction += currpos[est].predict_proba(molecules)[:,1]
                else:
                    prediction += currpos[est].predict(molecules)
            elif est == "Flat":
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
        
        # ec_index_dict = pickle_load('../utils/ec_index_dict.pkl')
        ec_index_dict = ['1.1.1.1', '1.1.3.10', '2.5.1.47']
        
        self.enzymes_tree = {}
        for enzyme in ec_index_dict:
            if type(enzyme) == str: # only pick up EC Numbers, not the integer indices
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
