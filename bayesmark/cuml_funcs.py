from bayesmark.sklearn_funcs import SklearnModel
import cupy as cp
from functools import partial

from cuml.metrics.regression import mean_absolute_error, mean_squared_error
from cuml.metrics.accuracy import accuracy_score
from cuml.metrics import log_loss

def cuml_get_scorer(metric, estimator, X, y):
    
    if metric == 'mae':
        yp = estimator.predict(X)
        return -mean_absolute_error(y, yp)
    elif metric == 'mse':
        yp = estimator.predict(X)
        return -mean_squared_error(y, yp)
    elif metric == 'nll':
        yp = estimator.predict_proba(X)
        return -log_loss(y, yp)
    elif metric == 'acc':
        yp = estimator.predict(X)
        return accuracy_score(y, yp)
    else:
        assert 0, "unknown metric"
        
def cuml_cross_val_score(clf, X, y, scoring, cv, n_jobs):
    ids = cp.arange(X.shape[0])
    cp.random.shuffle(ids)
    score = []
    for i in range(cv):
        train_index = ids%cv != i
        test_index = ids%cv == i
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf.fit(X_train, y_train)
        score.append(scoring(clf, X_test, y_test))
    return score

class CumlModel(SklearnModel):
            
    
    def __init__(self, model, dataset, metric, shuffle_seed=0, data_root=None):
        super().__init__(model, dataset, metric,
                         shuffle_seed=shuffle_seed, data_root=data_root)
        self.data_X = cp.asarray(self.data_X)
        self.data_Xt = cp.asarray(self.data_Xt)
        self.data_y  = cp.asarray(self.data_y)
        self.data_yt = cp.asarray(self.data_yt)

        self.scorer = partial(cuml_get_scorer, metric)
        self.n_jobs = 1
        self.cv_score = cuml_cross_val_score