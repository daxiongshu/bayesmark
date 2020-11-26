# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Routines to build a standardized interface to make `sklearn` hyper-parameter tuning problems look like an objective
function.

This file mostly contains a dictionary collection of all sklearn test funcs.

The format of each element in `MODELS` is:
model_name: (model_class, fixed_param_dict, search_param_api_dict)
`model_name` is an arbitrary name to refer to a certain strategy.
At usage time, the optimizer instance is created using:
``model_class(**kwarg_dict)``
The kwarg dict is `fixed_param_dict` + `search_param_dict`. The
`search_param_dict` comes from a optimizer which is configured using the
`search_param_api_dict`. See the API description for information on setting up
the `search_param_api_dict`.
"""
import warnings
warnings.filterwarnings("ignore")

import os.path
import pickle as pkl
import warnings
from abc import ABC, abstractmethod
from time import time

import numpy as np

from cuml.svm import SVC as cumlSVC
from cuml.svm import SVR as cumlSVR
from cuml import LogisticRegression as cumlLogisticRegression
from cuml import Ridge as cumlRidge
from cuml import Lasso as cumlLasso

from cuml.neighbors import KNeighborsClassifier as cumlKNeighborsClassifier
from cuml.neighbors import KNeighborsRegressor as cumlKNeighborsRegressor
from cuml.ensemble import RandomForestClassifier as cumlRandomForestClassifier
from cuml.ensemble import RandomForestRegressor as cumlRandomForestRegressor 

from bayesmark.MLP_cuml import MLPClassifier as cumlMLPClassifier
from bayesmark.MLP_cuml import MLPRegressor as cumlMLPRegressor

from bayesmark.xgb_cuml import XGBClassifier as cumlXGBClassifier
from bayesmark.xgb_cuml import XGBRegressor as cumlXGBRegressor

from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBRegressor, XGBClassifier

from bayesmark.constants import ARG_DELIM, METRICS, MODEL_NAMES, VISIBLE_TO_OPT
from bayesmark.data import METRICS_LOOKUP, ProblemType, get_problem_type, load_data
from bayesmark.path_util import absopen
from bayesmark.space import JointSpace
from bayesmark.util import str_join_safe

# Using 3 would be faster, but 5 is the most realistic CV split (5-fold)
CV_SPLITS = 5

def without(d, bad):
    return {k:v for k,v in d.items() if k!=bad and k not in bad}

# We should add cat variables into some of these configurations but a lot of
# the wrappers for the BO methods really have trouble with cat types.

# kNN
knn_cfg = {
    "n_neighbors": {"type": "int", "space": "linear", "range": (1, 25)},
    "p": {"type": "int", "space": "linear", "range": (1, 4)},
}

# SVM
svm_cfg = {
    "C": {"type": "real", "space": "log", "range": (1.0, 1e3)},
    "gamma": {"type": "real", "space": "log", "range": (1e-4, 1e-3)},
    "tol": {"type": "real", "space": "log", "range": (1e-5, 1e-1)},
}

# DT
dt_cfg = {
    "max_depth": {"type": "int", "space": "linear", "range": (1, 15)},
    "min_samples_split": {"type": "real", "space": "logit", "range": (0.01, 0.99)},
    "min_samples_leaf": {"type": "real", "space": "logit", "range": (0.01, 0.49)},
    "min_weight_fraction_leaf": {"type": "real", "space": "logit", "range": (0.01, 0.49)},
    "max_features": {"type": "real", "space": "logit", "range": (0.01, 0.99)},
    "min_impurity_decrease": {"type": "real", "space": "linear", "range": (0.0, 0.5)},
}

# RF
rf_cfg = {
    "max_depth": {"type": "int", "space": "linear", "range": (1, 15)},
    "max_features": {"type": "real", "space": "logit", "range": (0.01, 0.99)},
    "min_samples_split": {"type": "real", "space": "logit", "range": (0.01, 0.99)},
    "min_samples_leaf": {"type": "real", "space": "logit", "range": (0.01, 0.49)},
    "min_weight_fraction_leaf": {"type": "real", "space": "logit", "range": (0.01, 0.49)},
    "min_impurity_decrease": {"type": "real", "space": "linear", "range": (0.0, 0.5)},
}

# MLP with ADAM
mlp_adam_cfg = {
    "hidden_layer_sizes": {"type": "int", "space": "linear", "range": (50, 200)},
    "alpha": {"type": "real", "space": "log", "range": (1e-5, 1e1)},
    "batch_size": {"type": "int", "space": "linear", "range": (10, 250)},
    "learning_rate_init": {"type": "real", "space": "log", "range": (1e-5, 1e-1)},
    "tol": {"type": "real", "space": "log", "range": (1e-5, 1e-1)},
    "validation_fraction": {"type": "real", "space": "logit", "range": (0.1, 0.9)},
    "beta_1": {"type": "real", "space": "logit", "range": (0.5, 0.99)},
    "beta_2": {"type": "real", "space": "logit", "range": (0.9, 1.0 - 1e-6)},
    "epsilon": {"type": "real", "space": "log", "range": (1e-9, 1e-6)},
}

cuml_mlp_adam_cfg = mlp_adam_cfg.copy()
cuml_mlp_adam_cfg["batch_size"] = {"type": "int", "space": "linear", "range": (256, 1024)}

# MLP with SGD
mlp_sgd_cfg = {
    "hidden_layer_sizes": {"type": "int", "space": "linear", "range": (50, 200)},
    "alpha": {"type": "real", "space": "log", "range": (1e-5, 1e1)},
    "batch_size": {"type": "int", "space": "linear", "range": (10, 250)},
    "learning_rate_init": {"type": "real", "space": "log", "range": (1e-5, 1e-1)},
    "power_t": {"type": "real", "space": "logit", "range": (0.1, 0.9)},
    "tol": {"type": "real", "space": "log", "range": (1e-5, 1e-1)},
    "momentum": {"type": "real", "space": "logit", "range": (0.001, 0.999)},
    "validation_fraction": {"type": "real", "space": "logit", "range": (0.1, 0.9)},
}

cuml_mlp_sgd_cfg = mlp_sgd_cfg.copy()
cuml_mlp_sgd_cfg["batch_size"] = {"type": "int", "space": "linear", "range": (256, 1024)}

# AdaBoostClassifier
ada_cfg = {
    "n_estimators": {"type": "int", "space": "linear", "range": (10, 100)},
    "learning_rate": {"type": "real", "space": "log", "range": (1e-4, 1e1)},
}

# lasso
lasso_cfg = {
    "C": {"type": "real", "space": "log", "range": (1e-2, 1e2)},
    "intercept_scaling": {"type": "real", "space": "log", "range": (1e-2, 1e2)},
}

# linear
linear_cfg = {
    "C": {"type": "real", "space": "log", "range": (1e-2, 1e2)},
    "intercept_scaling": {"type": "real", "space": "log", "range": (1e-2, 1e2)},
}

# xgb
xgb_cfg = {
    #'n_estimators': {"type": "int", "space": "log", "range": (10, 1000)},
    'max_depth': {"type": "int", "space": "linear", "range": (1, 10)},
    'learning_rate': {"type": "real", "space": "linear", "range": (0.05, 0.5)},
    'subsample': {"type": "real", "space": "linear", "range": (0.1, 1.0)},
    'colsample_bytree': {"type": "real", "space": "linear", "range": (0.1, 1.0)},
}

MODELS_CLF = {
    "kNN": (KNeighborsClassifier, {}, knn_cfg),
    "kNN-cuml": (cumlKNeighborsClassifier, {}, without(knn_cfg, ['p'])),
    "SVM": (SVC, {"kernel": "rbf", "probability": True}, svm_cfg),
    "SVM-cuml": (cumlSVC, {"kernel": "rbf", "probability": True}, svm_cfg),
    "DT": (DecisionTreeClassifier, {"max_leaf_nodes": None}, dt_cfg),
    "RF": (RandomForestClassifier, {"n_estimators": 10, "max_leaf_nodes": None}, rf_cfg),
    "RF-cuml": (cumlRandomForestClassifier, {"n_estimators": 10, "max_leaves": -1}, without(rf_cfg, ["min_samples_split", "min_samples_leaf", "min_weight_fraction_leaf"])),
    "MLP-adam": (MLPClassifier, {"solver": "adam", "early_stopping": True}, mlp_adam_cfg),
    "MLP-adam-cuml": (cumlMLPClassifier, {"solver": "adam", "early_stopping": True}, cuml_mlp_adam_cfg),
    "MLP-sgd": (
        MLPClassifier,
        {"solver": "sgd", "early_stopping": True, "learning_rate": "invscaling", "nesterovs_momentum": True},
        mlp_sgd_cfg,
    ),
    "MLP-sgd-cuml": (
        cumlMLPClassifier,
        {"solver": "sgd", "early_stopping": True, "learning_rate": "invscaling", "nesterovs_momentum": True},
        cuml_mlp_sgd_cfg,
    ),
    "ada": (AdaBoostClassifier, {}, ada_cfg),
    "lasso": (
        LogisticRegression,
        {"penalty": "l1", "fit_intercept": True, "solver": "liblinear", "multi_class": "ovr"},
        lasso_cfg,
    ),
    "lasso-cuml": (
        cumlLogisticRegression,
        {"penalty": "l1", "fit_intercept": True, "solver": "qn", "max_iter":100},
        without(lasso_cfg, ["intercept_scaling"]),
    ),
    "linear": (
        LogisticRegression,
        {"penalty": "l2", "fit_intercept": True, "solver": "liblinear", "multi_class": "ovr"},
        linear_cfg,
    ),
    "linear-cuml": (
        cumlLogisticRegression,
        {"penalty": "l2", "fit_intercept": True, "solver": "qn", "max_iter":100},
        without(linear_cfg, ["intercept_scaling"]),
    ),
 
    "xgb": (XGBClassifier, {'n_estimators': 100,
                            'validation_fraction': 0
                           }, xgb_cfg),
    "xgb-cuml": (cumlXGBClassifier, {'tree_method': 'gpu_hist', 
                                 'predictor': 'gpu_predictor',
                                 'n_estimators': 100,
                                 'validation_fraction': 0
                                }, xgb_cfg),
}

# For now, we will assume the default is to go thru all classifiers
assert sorted(MODELS_CLF.keys()) == sorted(MODEL_NAMES)

ada_cfg_reg = {
    "n_estimators": {"type": "int", "space": "linear", "range": (10, 100)},
    "learning_rate": {"type": "real", "space": "log", "range": (1e-4, 1e1)},
}

lasso_cfg_reg = {
    "alpha": {"type": "real", "space": "log", "range": (1e-2, 1e2)},
    "fit_intercept": {"type": "bool"},
    "normalize": {"type": "bool"},
    "max_iter": {"type": "int", "space": "log", "range": (10, 5000)},
    "tol": {"type": "real", "space": "log", "range": (1e-5, 1e-1)},
    "positive": {"type": "bool"},
}

linear_cfg_reg = {
    "alpha": {"type": "real", "space": "log", "range": (1e-2, 1e2)},
    "fit_intercept": {"type": "bool"},
    "normalize": {"type": "bool"},
    "max_iter": {"type": "int", "space": "log", "range": (10, 5000)},
    "tol": {"type": "real", "space": "log", "range": (1e-4, 1e-1)},
}

MODELS_REG = {
    "kNN": (KNeighborsRegressor, {}, knn_cfg),
    "kNN-cuml": (cumlKNeighborsRegressor, {}, without(knn_cfg, ['p'])),
    "SVM": (SVR, {"kernel": "rbf"}, svm_cfg),
    "SVM-cuml": (cumlSVR, {"kernel": "rbf", 'nochange_steps':1, 'max_iter':1000}, svm_cfg),
    "DT": (DecisionTreeRegressor, {"max_leaf_nodes": None}, dt_cfg),
    "RF": (RandomForestRegressor, {"n_estimators": 10, "max_leaf_nodes": None}, rf_cfg),
    "RF-cuml": (cumlRandomForestRegressor, {"n_estimators": 10, "max_leaves": -1}, without(rf_cfg, ["min_samples_split", "min_samples_leaf", "min_weight_fraction_leaf"])),
    "MLP-adam": (MLPRegressor, {"solver": "adam", "early_stopping": True}, mlp_adam_cfg),
    "MLP-adam-cuml": (cumlMLPRegressor, {"solver": "adam", "early_stopping": True}, cuml_mlp_adam_cfg),
    "MLP-sgd": (
        MLPRegressor,  # regression crashes often with relu
        {
            "activation": "tanh",
            "solver": "sgd",
            "early_stopping": True,
            "learning_rate": "invscaling",
            "nesterovs_momentum": True,
        },
        mlp_sgd_cfg,
    ),
    "MLP-sgd-cuml": (
        cumlMLPRegressor,  # regression crashes often with relu
        {
            "activation": "tanh",
            "solver": "sgd",
            "early_stopping": True,
            "learning_rate": "invscaling",
            "nesterovs_momentum": True,
        },
        cuml_mlp_sgd_cfg,
    ),
    "ada": (AdaBoostRegressor, {}, ada_cfg_reg),
    "lasso": (Lasso, {}, lasso_cfg_reg),
    "lasso-cuml": (cumlLasso, {}, without(lasso_cfg_reg, ['positive'])),
    "linear": (Ridge, {"solver": "auto"}, linear_cfg_reg),
    "linear-cuml": (cumlRidge, {}, without(linear_cfg_reg, ['max_iter', 'tol'])),
    
    "xgb": (XGBRegressor, {'n_estimators': 100,
                           'validation_fraction': 0
                          }, xgb_cfg),
    "xgb-cuml": (cumlXGBRegressor, {'tree_method': 'gpu_hist', 
                                'predictor': 'gpu_predictor',
                                'n_estimators': 100,
                                'validation_fraction': 0,
                               }, xgb_cfg),
}

# If both classifiers and regressors match MODEL_NAMES then the experiment
# launcher can simply go thru the cartesian product and do all combos.
assert sorted(MODELS_REG.keys()) == sorted(MODEL_NAMES)


class TestFunction(ABC):
    """Abstract base class for test functions in the benchmark. These do not need to be ML hyper-parameter tuning.
    """

    def __init__(self):
        """Setup general test function for benchmark. We assume the test function knows the meta-data about the search
        space, but is also stateless to fit modeling assumptions. To keep stateless, it does not do things like count
        the number of function evaluations.
        """
        # This will need to be set before using other routines
        self.api_config = None

    @abstractmethod
    def evaluate(self, params):
        """Abstract method to evaluate the function at a parameter setting.
        """

    def get_api_config(self):
        """Get the API config for this test problem.

        Returns
        -------
        api_config : dict(str, dict(str, object))
            The API config for the used model. See README for API description.
        """
        assert self.api_config is not None, "API config is not set."
        return self.api_config


class SklearnModel(TestFunction):
    """Test class for sklearn classifier/regressor CV score objective functions.
    """

    # Map our short names for metrics to the full length sklearn name
    _METRIC_MAP = {
        "nll": "neg_log_loss",
        "acc": "accuracy",
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
    }

    # This can be static and constant for now
    objective_names = (VISIBLE_TO_OPT, "generalization")

    def __init__(self, model, dataset, metric, shuffle_seed=0, data_root=None, n_jobs=None):
        """Build class that wraps sklearn classifier/regressor CV score for use as an objective function.

        Parameters
        ----------
        model : str
            Which classifier to use, must be key in `MODELS_CLF` or `MODELS_REG` dict depending on if dataset is
            classification or regression.
        dataset : str
            Which data set to use, must be key in `DATA_LOADERS` dict, or name of custom csv file.
        metric : str
            Which sklearn scoring metric to use, in `SCORERS_CLF` list or `SCORERS_REG` dict depending on if dataset is
            classification or regression.
        shuffle_seed : int
            Random seed to use when splitting the data into train and validation in the cross-validation splits. This
            is needed in order to keep the split constant across calls. Otherwise there would be extra noise in the
            objective function for varying splits.
        data_root : str
            Root directory to look for all custom csv files.
        """
        TestFunction.__init__(self)
        data, target, problem_type = load_data(dataset, data_root=data_root)
        assert problem_type in (ProblemType.clf, ProblemType.reg)
        self.is_classifier = problem_type == ProblemType.clf

        # Do some validation on loaded data
        assert isinstance(data, np.ndarray)
        assert isinstance(target, np.ndarray)
        assert data.ndim == 2 and target.ndim == 1
        assert data.shape[0] == target.shape[0]
        assert data.size > 0
        assert data.dtype == np.float_
        assert np.all(np.isfinite(data))  # also catch nan
        assert target.dtype == (np.int_ if self.is_classifier else np.float_)
        assert np.all(np.isfinite(target))  # also catch nan

        model_lookup = MODELS_CLF if self.is_classifier else MODELS_REG
        base_model, fixed_params, api_config = model_lookup[model]

        # New members for model
        self.base_model = base_model
        self.fixed_params = fixed_params
        self.api_config = api_config

        # Always shuffle your data to be safe. Use fixed seed for reprod.
        self.data_X, self.data_Xt, self.data_y, self.data_yt = train_test_split(
            data, target, test_size=0.2, random_state=shuffle_seed, shuffle=True
        )

        assert metric in METRICS, "Unknown metric %s" % metric
        assert metric in METRICS_LOOKUP[problem_type], "Incompatible metric %s with problem type %s" % (
            metric,
            problem_type,
        )
        self.scorer = get_scorer(SklearnModel._METRIC_MAP[metric])
        self.n_jobs = n_jobs
        self.cv_score = cross_val_score

    def evaluate(self, params):
        """Evaluate the sklearn CV objective at a particular parameter setting.

        Parameters
        ----------
        params : dict(str, object)
            The varying (non-fixed) parameter dict to the sklearn model.

        Returns
        -------
        cv_loss : float
            Average loss over CV splits for sklearn model when tested using the settings in params.
        """
        start = time()
        params = dict(params)  # copy to avoid modification of original
        params.update(self.fixed_params)  # add in fixed params

        # now build the skl object
        clf = self.base_model(**params)

        assert np.all(np.isfinite(self.data_X)), "all features must be finite"
        assert np.all(np.isfinite(self.data_y)), "all targets must be finite"

        s1 = time()
        # Do the x-val, ignore user warn since we expect BO to try weird stuff
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            S = self.cv_score(clf, self.data_X, self.data_y, scoring=self.scorer, cv=CV_SPLITS, n_jobs=self.n_jobs)
        d1 = time() - s1
        # Take the mean score across all x-val splits
        cv_score = np.mean(S)

        # Now let's get the generalization error for same hypers
        clf = self.base_model(**params)
        clf.fit(self.data_X, self.data_y)
        generalization_score = self.scorer(clf, self.data_Xt, self.data_yt)

        # get_scorer makes everything a score not a loss, so we need to negate to get the loss back
        cv_loss = -cv_score
        assert np.isfinite(cv_loss), "loss not even finite"
        generalization_loss = -generalization_score
        assert np.isfinite(generalization_loss), "loss not even finite"

        # Unbox to basic float to keep it simple
        cv_loss = cv_loss.item()
        assert isinstance(cv_loss, float)
        if hasattr(generalization_loss, 'item'):
            generalization_loss = generalization_loss.item()
        assert isinstance(generalization_loss, float)

        # For now, score with same objective. We can later add generalization error
        duration = time() - start
        #print(f"eval duration: {duration:.1f} seconds cv: {d1:.1f} seconds")
        return cv_loss, generalization_loss

    @staticmethod
    def test_case_str(model, dataset, scorer):
        """Generate the combined test case string from model, dataset, and scorer combination."""
        test_case = str_join_safe(ARG_DELIM, (model, dataset, scorer))
        return test_case

    @staticmethod
    def inverse_test_case_str(test_case):
        """Inverse of `test_case_str`."""
        model, dataset, scorer = test_case.split(ARG_DELIM)
        assert test_case == SklearnModel.test_case_str(model, dataset, scorer)
        return model, dataset, scorer


class SklearnSurrogate(TestFunction):
    """Test class for sklearn classifier/regressor CV score objective function surrogates.
    """

    # This can be static and constant for now
    objective_names = (VISIBLE_TO_OPT, "generalization")

    def __init__(self, model, dataset, scorer, path):
        """Build class that wraps sklearn classifier/regressor CV score for use as an objective function surrogate.

        Parameters
        ----------
        model : str
            Which classifier to use, must be key in `MODELS_CLF` or `MODELS_REG` dict depending on if dataset is
            classification or regression.
        dataset : str
            Which data set to use, must be key in `DATA_LOADERS` dict, or name of custom csv file.
        scorer : str
            Which sklearn scoring metric to use, in `SCORERS_CLF` list or `SCORERS_REG` dict depending on if dataset is
            classification or regression.
        path : str
            Root directory to look for all pickle files.
        """
        TestFunction.__init__(self)

        # Find the space class, we could consider putting this in pkl too
        problem_type = get_problem_type(dataset)
        assert problem_type in (ProblemType.clf, ProblemType.reg)
        _, _, self.api_config = MODELS_CLF[model] if problem_type == ProblemType.clf else MODELS_REG[model]
        self.space = JointSpace(self.api_config)

        # Load the pre-trained model
        fname = SklearnModel.test_case_str(model, dataset, scorer) + ".pkl"

        if isinstance(path, bytes):
            # This is for test-ability, we could use mock instead.
            self.model = pkl.loads(path)
        else:
            path = os.path.join(path, fname)  # pragma: io
            assert os.path.isfile(path), "Model file not found: %s" % path

            with absopen(path, "rb") as f:  # pragma: io
                self.model = pkl.load(f)  # pragma: io
        assert callable(getattr(self.model, "predict", None))

    def evaluate(self, params):
        """Evaluate the sklearn CV objective at a particular parameter setting.

        Parameters
        ----------
        params : dict(str, object)
            The varying (non-fixed) parameter dict to the sklearn model.

        Returns
        -------
        overall_loss : float
            Average loss over CV splits for sklearn model when tested using the settings in params.
        """
        x = self.space.warp([params])
        y, = self.model.predict(x)

        assert y.shape == (len(self.objective_names),)
        assert y.dtype.kind == "f"

        assert np.all(-np.inf < y)  # Will catch nan too
        y = tuple(y.tolist())  # Make consistent with SklearnModel typing
        return y
