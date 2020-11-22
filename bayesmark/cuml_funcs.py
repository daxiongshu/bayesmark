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