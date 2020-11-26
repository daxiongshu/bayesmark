from bayesmark.sklearn_funcs import SklearnModel
import cupy as cp
from functools import partial

from cuml.metrics.regression import mean_absolute_error, mean_squared_error
from cuml.metrics.accuracy import accuracy_score
from cuml.metrics import log_loss


def cuml_get_scorer(metric, estimator, X, y):
    yp = estimator.predict_proba(X) if metric == 'nll' else estimator.predict(X)
    if isinstance(yp, cp.ndarray) == False:
        yp = cp.asarray(yp)
    if metric == 'mae':
        return -mean_absolute_error(y, yp)
    elif metric == 'mse':
        return -mean_squared_error(y, yp)
    elif metric == 'nll':
        return -log_loss(y, yp)
    elif metric == 'acc':
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
        s = scoring(clf, X_test, y_test)
        if hasattr(s, 'item'):
            s = s.item()
        score.append(s)
    return score

class Normalizer:
    def __init__(self):
        pass

    def fit(self, X):
        self.mean = cp.mean(X, axis=0, keepdims=True)
        self.std = cp.std(X, axis=0, keepdims=True)
        return self

    def transform(self, X):
        return (X - self.mean)/self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class MinMax:
    def __init__(self):
        pass

    def fit(self, X):
        self.min_ = cp.min(X)#, axis=0, keepdims=True)
        self.max_ = cp.max(X)#, axis=0, keepdims=True)
        return self

    def transform(self, X):
        return (X - self.min_)/(self.max_ - self.min_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return X*(self.max_ - self.min_) + self.min_

class CumlModel(SklearnModel):
            
    
    def __init__(self, model, dataset, metric, shuffle_seed=0, data_root=None):
        super().__init__(model, dataset, metric,
                         shuffle_seed=shuffle_seed, data_root=data_root)
        self.data_X = cp.asarray(self.data_X, dtype='float32')
        self.data_Xt = cp.asarray(self.data_Xt, dtype='float32')
        dtype = 'int32' if metric in ['nll', 'acc'] else 'float32'
        self.data_y  = cp.asarray(self.data_y, dtype=dtype)
        self.data_yt = cp.asarray(self.data_yt, dtype=dtype)

        if model == 'SVM-cuml':
            self.norm = Normalizer()
            self.data_X = self.norm.fit_transform(self.data_X)
            self.data_Xt = self.norm.transform(self.data_Xt)

            self.mm = MinMax()
            self.data_y = self.mm.fit_transform(self.data_y)
            self.data_yt = self.mm.transform(self.data_yt)

        self.scorer = partial(cuml_get_scorer, metric)
        self.n_jobs = 1
        self.cv_score = cuml_cross_val_score
