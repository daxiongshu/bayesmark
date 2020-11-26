import xgboost as xgb
import cupy as cp
from cuml.preprocessing.model_selection import train_test_split 

def get_default_params():
    return  {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'tree_method': 'gpu_hist',
            'verbosity': 0,
            'early_stopping':False,
            'validation_fraction':0.1,
            'early_stopping_rounds':10,
        }

def print_shape(*x):
    for i in x:
        print(i.shape, end=' ')
    print()

class XGBbase:
    
    def __init__(self, **params):
        self.params = get_default_params()
        self.params.update(params)
        
    def fit(self, X, y):
        
        test_size = self.params['validation_fraction']
        num_boost_round = self.params['n_estimators']
        
        #ext_params = ['early_stopping', 'early_stopping_rounds', 
        #              'n_estimators', 'silent', 'validation_fraction', 'verbose']
        
        if test_size >0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
            dtrain = xgb.DMatrix(data=X_train, label=y_train)
            dvalid = xgb.DMatrix(data=X_test, label=y_test)
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')] 
            early_stopping_rounds = self.params['early_stopping_rounds']
            self.clf = xgb.train(self.params, dtrain=dtrain,
                num_boost_round=num_boost_round,evals=watchlist,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=1000)
        else:
            dtrain = xgb.DMatrix(data=X, label=y)
            self.clf = xgb.train(self.params, dtrain=dtrain,
                num_boost_round=num_boost_round)

        return self
    
    def predict(self, X):
        self.clf.set_param({'predictor': 'gpu_predictor'})
        dtest = xgb.DMatrix(data=X)
        yp = self.clf.predict(dtest)
        yp = cp.asarray(yp)
        return yp
    
class XGBClassifier(XGBbase):
    
    def __init__(self, **params):
        super().__init__(**params)
        
    def fit(self, X, y):
        num_class = int(y.max()+1)
        if num_class > 2:
            self.params['num_class'] = num_class
        else:
            self.params['objective'] = 'binary:logistic'
            self.params['eval_metric'] = 'logloss'
        return super().fit(X, y)
        
    def predict_proba(self, X):
        return super().predict(X)
    
    def predict(self, X):
        yp = super().predict(X)
        if len(yp.shape) == 2:
            yp = cp.argmax(yp, axis=1)
        else:
            yp = yp>0.5
        return yp
    
class XGBRegressor(XGBbase):
    
    def __init__(self, **params):
        params['objective'] = 'reg:squarederror'
        params['eval_metric'] = 'rmse'
        super().__init__(**params)
