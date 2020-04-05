import numpy as np
import sklearn

class GridSearchCV(object):
    
    def __init__(self,
                 estimator,
                 param_grid,
                 cv=5,
                 proba=False):
        
        self.estimator = estimator
        self.num_folds = cv
        self.param_grid = param_grid
        self.proba = proba
        
    def prepare_folds(self, x_NF, y_N, y_prev, y_binary_orig, sample_weight):
        x_pos_NF = x_NF[y_binary_orig == 1.0]
        y_pos_N = y_N[y_binary_orig == 1.0]
        y_pos_prev = y_prev[y_binary_orig == 1.0]
        y_pos_binary_orig = y_binary_orig[y_binary_orig == 1.0]
        sample_weight_pos = sample_weight[y_binary_orig == 1.0]
        
        x_neg_NF = x_NF[y_binary_orig == 0.0]
        y_neg_N = y_N[y_binary_orig == 0.0]
        y_neg_prev = y_prev[y_binary_orig == 0.0]
        y_neg_binary_orig = y_binary_orig[y_binary_orig == 0.0]
        sample_weight_neg = sample_weight[y_binary_orig == 0.0]
        
        x_tr_NF_list_pos, y_tr_N_list_pos, w_tr_N_list_pos, x_va_NF_list_pos, y_va_N_list_pos, y_va_prev_list_pos, y_va_orig_list_pos, w_va_N_list_pos = self.prepare_singleclass_folds(x_pos_NF, y_pos_N, y_pos_prev, y_pos_binary_orig, sample_weight_pos)
        
        x_tr_NF_list_neg, y_tr_N_list_neg, w_tr_N_list_neg, x_va_NF_list_neg, y_va_N_list_neg, y_va_prev_list_neg, y_va_orig_list_neg, w_va_N_list_neg = self.prepare_singleclass_folds(x_neg_NF, y_neg_N, y_neg_prev, y_neg_binary_orig, sample_weight_neg)
        
        x_tr_NF_list = []
        y_tr_N_list = []
        w_tr_N_list = []
        x_va_NF_list = []
        y_va_N_list = []
        y_va_prev_list = []
        y_va_orig_list = []
        w_va_N_list = []
        for i in range(self.num_folds):
            x_tr_NF_list.append(np.vstack((x_tr_NF_list_pos[i], x_tr_NF_list_neg[i])))
            y_tr_N_list.append(np.hstack((y_tr_N_list_pos[i], y_tr_N_list_neg[i])))
            w_tr_N_list.append(np.hstack((w_tr_N_list_pos[i], w_tr_N_list_neg[i])))
            x_va_NF_list.append(np.vstack((x_va_NF_list_pos[i], x_va_NF_list_neg[i])))
            y_va_N_list.append(np.hstack((y_va_N_list_pos[i], y_va_N_list_neg[i])))
            y_va_prev_list.append(np.hstack((y_va_prev_list_pos[i], y_va_prev_list_neg[i])))
            y_va_orig_list.append(np.hstack((y_va_orig_list_pos[i], y_va_orig_list_neg[i])))
            w_va_N_list.append(np.hstack((w_va_N_list_pos[i], w_va_N_list_neg[i])))
       
        return x_tr_NF_list, y_tr_N_list, w_tr_N_list, x_va_NF_list, y_va_N_list, y_va_prev_list, y_va_orig_list, w_va_N_list
        
    def prepare_singleclass_folds(self, x_NF, y_N, y_prev, y_binary_orig, sample_weight):
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
        y_va_prev_list = []
        y_va_orig_list = []
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
            y_va_prev = y_prev[fold_start:fold_stop].copy()
            y_va_orig = y_binary_orig[fold_start:fold_stop].copy()
            w_va_N = sample_weight[fold_start:fold_stop].copy()
            
            x_tr_NF_list.append(x_tr_NF)
            y_tr_N_list.append(y_tr_N)
            w_tr_N_list.append(w_tr_N)
            x_va_NF_list.append(x_va_NF)
            y_va_N_list.append(y_va_N)
            y_va_prev_list.append(y_va_prev)
            y_va_orig_list.append(y_va_orig)
            w_va_N_list.append(w_va_N)
        
        return x_tr_NF_list, y_tr_N_list, w_tr_N_list, x_va_NF_list, y_va_N_list, y_va_prev_list, y_va_orig_list, w_va_N_list
    
    def fit(self, x_NF, y_N, y_prev, y_binary_orig, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(y_N.size)
        
        x_tr_NF_list, y_tr_N_list, w_tr_N_list, x_va_NF_list, y_va_N_list, y_prev_list, y_va_orig_list, w_va_N_list = self.prepare_folds(x_NF, y_N, y_prev, y_binary_orig, sample_weight)       
        
        scores = []

        param_combinations = generate_argument_dicts(self.param_grid)

        for params in param_combinations:
            param_scores = []
            for fold in range(self.num_folds):
                self.estimator.set_params(**params)
                self.estimator.fit(x_tr_NF_list[fold], y_tr_N_list[fold], sample_weight=w_tr_N_list[fold])
                s = self.average_precision_hierarchy(self.estimator, x_va_NF_list[fold], y_prev_list[fold], y_va_orig_list[fold], w_va_N_list[fold])
                param_scores.append(s)
            scores.append(np.mean(param_scores))
        
        best_params = param_combinations[np.argmax(np.array(scores))]
        
        self.estimator.set_params(**best_params)
        self.estimator.fit(x_NF, y_N)
        
        self.best_estimator_ = self.estimator
        self.best_params_ = best_params
        self.best_score_ = np.max(np.array(scores))
        
    def average_precision_hierarchy(self, estimator, x, y_prev, y_binary_orig, sample_weight):
        if self.proba:
            y_hat = estimator.predict_proba(x)[:,1]
        else:
            y_hat = estimator.predict(x)
        y_hat += y_prev

        # y_hat[y_hat >= 0.5] = 1.0
        # y_hat[y_hat < 0.5] = 0.0
        
        score = sklearn.metrics.average_precision_score(y_binary_orig, y_hat, sample_weight=sample_weight)
        
        return score
        
def balanced_accuracy_hierarchy(estimator, x, y_prev, y_binary_orig, sample_weight):
    y_hat = estimator.predict(x) + y_prev
    
    y_hat[y_hat >= 0.5] = 1.0
    y_hat[y_hat < 0.5] = 0.0
#     print(y_hat)
#     print(y_binary_orig)
    
    score = sklearn.metrics.balanced_accuracy_score(y_binary_orig, y_hat, sample_weight=sample_weight)
    
    return score

            
def generate_argument_dicts(param_grid):
    params = []
    values_list = []
    for param in param_grid:
        params.append(param)
        values_list.append(param_grid[param])  
    list_of_permutations = [list(x) for x in np.array(np.meshgrid(*values_list)).T.reshape(-1,len(values_list))]
    
    dicts = []
    for perms in list_of_permutations:
        adict = {}
        for i, value in enumerate(perms):
            if value != 'sqrt':
                value = int(value)
            adict[params[i]] = value
                
        dicts.append(adict)
    return dicts
                
            
        
        
        
        
        
        
        
        
        
        
        
        