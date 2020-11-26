import torch
import cupy as cp
from torch.utils.data import Dataset, DataLoader
from torch.utils.dlpack import from_dlpack,to_dlpack
from cuml.preprocessing.model_selection import train_test_split 
import time

from sklearn.datasets import load_boston, load_breast_cancer, load_iris, load_digits
import cupy as cp
import numpy as np
from cuml.metrics import log_loss
from cuml.metrics.regression import mean_absolute_error, mean_squared_error


class MyDataSet(Dataset):

    def __init__(self, X, y=None, task='classification'):
        
        self.X = X
        self.y = y if y is not None else cp.zeros(X.shape[0])
        self.task = task

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y

    def getitems(self, ids):
        # The first column is label
        x = self.X[ids]
        x = from_dlpack(x.toDlpack()).float()
        
        y = self.y[ids]
        y = from_dlpack(y.toDlpack())
        y = y.float() if self.task == 'regression' else y.long()
        return x, y
    
class MyDataLoader:

    def __init__(self, dataset, batch_size, shuffle = True, drop_last=True):
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.dataset = dataset
        self.num_samples = len(self.dataset)
        self.batch_size = min(batch_size, self.num_samples)
        
    def __iter__(self):
        self.n = 0
        self.ids = cp.arange(self.num_samples)
        if self.shuffle:
            cp.random.shuffle(self.ids)
        return self
    
    def __len__(self):
        return len(self.dataset)
    
    @property
    def total_batches(self):
        return self.num_samples//self.batch_size + 1

    def get_batch(self,ids):
        x,y = self.dataset.getitems(ids)
        if self.dataset.task == 'regression':
            y = y.unsqueeze(1)
        return x,y
    
    def __next__(self):
        if self.n < self.num_samples:
            if self.n + self.batch_size <= self.num_samples:
                ids = self.ids[self.n:self.n+self.batch_size]
                self.n += self.batch_size
                return self.get_batch(ids)
            elif self.drop_last == False:
                ids = self.ids[self.n:]
                self.n += self.batch_size
                return self.get_batch(ids)
            else:
                raise StopIteration
        else:
            raise StopIteration


class MLP(torch.nn.Module):
    
    def __init__(self, N, Hs, C, **params):
        
        super(MLP, self).__init__()
        self.params = params
        
        if hasattr(Hs, '__iter__'):
            Hs = list(Hs)
        else:
            Hs = [Hs]
        Hs = [N] + Hs + [C]
        
        layers = []
        for c,H in enumerate(Hs[:-1]):
            layer = torch.nn.Linear(in_features=H, out_features=Hs[c+1])
            layers.append(layer)
        self.layers = torch.nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = activate(x, self.params['activation'])
        layer = self.layers[-1]
        return layer(x)
    
def activate(x, act):
    if act == 'identity':
        return x
    elif act == 'relu':
        return torch.nn.functional.relu(x)
    elif act == 'tanh':
        return torch.nn.functional.tanh(x)
    elif act == 'sigmoid':
        return torch.nn.functional.sigmoid(x)
    else:
        assert 0, "unknown activation "+act


def get_default_params():
   return {
       'hidden_layer_sizes': (100, ), 
       'activation':'relu',
       'solver':'adam', 
       'alpha':0.0001,
       'batch_size':'auto', 
       'learning_rate':'constant', 
       'learning_rate_init':0.001, 
       'max_iter':200, 
       'shuffle':True, 
       'random_state':None, 
       'tol':0.0001, 
       'verbose':False,
       'momentum':0.9, 
       'nesterovs_momentum':True, 
       'early_stopping':False, 
       'validation_fraction':0.1, 
       'beta_1':0.9, 
       'beta_2':0.999, 
       'epsilon':1e-08, 
       'n_iter_no_change':10, 
       'max_fun':15000,
   }



class Learner(object):
    
    def __init__(self, model, loss_func, **params): 
        
        self.model = model
        self.loss_func = loss_func
        
        self.model.cuda()
        if params.get('multi_gpu',False):
            self.model = torch.nn.DataParallel(self.model)
        self.params = get_default_params()
        self.params.update(params)
        
    def predict(self,test_dl):
        yps = []
        for batch in test_dl:
            xb, _ = batch
            xb = xb.cuda()
            pred = self.model(xb)
            pred = to_dlpack(pred.detach())
            yps.append(cp.fromDlpack(pred))
        yps = cp.vstack(yps)
        if yps.shape[1] == 1:
            yps = yps.ravel()
        return yps
        
    def fit(self,train_dl,valid_dl=None):
        
        lr = self.params['learning_rate_init']
        epochs = self.params['max_iter']
        opt_type = self.params['solver']
        wd = self.params['alpha']
        momentum = self.params['momentum']
        beta1, beta2 = self.params['beta_1'], self.params['beta_2']
        eps = self.params['epsilon']
        
        if opt_type == 'sgd':
            opt = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd, momentum=momentum, nesterov=self.params['nesterovs_momentum'])
        elif opt_type == 'adam':
            opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd, betas=(beta1, beta2), eps=eps)
        else:
            opt = torch.optim.LBFGS(self.model.parameters(), lr=lr)
            
        best_score = 1e9 # lower the better  
        not_improved_iter = 0
        tol = self.params['tol']
        n_iter_no_change = self.params['n_iter_no_change']
        
        for epoch in range(epochs):
            train_loss = 0
 
            start = time.time()
    
            for batch in train_dl:
                
                xb, yb = batch
                xb, yb = xb.cuda(),yb.cuda()
                pred = self.model(xb)
                loss = self.loss_func(pred, yb)
                def closure():
                    return loss
                train_loss += loss.cpu().detach().numpy()
                loss.backward()
                if opt_type!='lbfgs':
                    opt.step()
                else:
                    opt.step(closure)
                opt.zero_grad()
            if valid_dl is None:
                duration = time.time() - start
                print('Epoch %d Training Loss:%.4f Time: %.2f seconds'%(epoch,
                            train_loss/train_dl.total_batches,duration))
                continue
                
            valid_loss = 0
            for batch in valid_dl:
                xb, yb = batch
                xb, yb = xb.cuda(),yb.cuda()
                pred = self.model(xb)
                loss = self.loss_func(pred, yb)
                valid_loss += loss.cpu().detach().numpy()
                
            valid_loss /= valid_dl.total_batches
            duration = time.time() - start
            if self.params['verbose']:
                print('Epoch %d Training Loss:%.4f Valid Loss:%4f Time: %.2f seconds'%(epoch,
                            train_loss/train_dl.total_batches, valid_loss, duration))
            
            if valid_loss < best_score - tol:
                best_score = valid_loss
            else:
                not_improved_iter += 1
                
            if self.params['early_stopping'] and not_improved_iter > n_iter_no_change:
                break


class MLPbase:
    
    def __init__(self, **params):
        self.params = get_default_params()
        self.params.update(params)
        self.task = None
        
    def get_batch_size(self, N, mb=1024):
        if self.params['batch_size'] == 'auto':
            return min(N, mb)
        return min(N, self.params['batch_size'])
    
    def fit(self, X, y):
        
        Xmean, Xstd = cp.mean(X, axis=0, keepdims=True), cp.std(X, axis=0, keepdims=True)
        X = (X - Xmean) / (Xstd + 1e-5)
        self.Xmean, self.Xstd = Xmean, Xstd
        
        test_size = self.params['validation_fraction']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
        
        train_dataset = MyDataSet(X_train, y_train, task=self.task)
        valid_dataset = MyDataSet(X_test, y_test, task=self.task)
        
        batch_size = self.get_batch_size(N=X_train.shape[0], mb=1024)

        train_dataloader = MyDataLoader(train_dataset, batch_size=batch_size,
                                shuffle=self.params['shuffle'], drop_last=True)

        batch_size = self.get_batch_size(N=X_test.shape[0], mb=1024)
        valid_dataloader = MyDataLoader(valid_dataset, batch_size=batch_size,
                                shuffle=False, drop_last=False)
        
        C = 1 if self.task == 'regression' else cp.unique(y).shape[0]
        model = MLP(X.shape[1], self.params['hidden_layer_sizes'], C, **self.params)
        
        if self.task == 'regression':
            loss_func = torch.nn.functional.mse_loss
        elif self.task == 'classification':
            loss_func = torch.nn.functional.cross_entropy
        else:
            assert 0, "Unknown taks: "+self.task
        
        learner = Learner(model, loss_func, **self.params)
        learner.fit(train_dl=train_dataloader, valid_dl=valid_dataloader)
        self.learner = learner
        
    def predict_(self, X):
        X = (X - self.Xmean) / (self.Xstd + 1e-5)
        test_dataset = MyDataSet(X)
        batch_size = self.get_batch_size(N=X.shape[0], mb=1024)
        test_dataloader = MyDataLoader(test_dataset, batch_size=batch_size,
                        shuffle=False, drop_last=False)
        
        yp = self.learner.predict(test_dataloader)
        return yp
    
    def predict(self, X):
        yp = self.predict_(X)
        if self.task == 'classification':
            yp = cp.argmax(yp, axis=1)
        return yp
    

class MLPRegressor(MLPbase):
    
    def __init__(self, **params):
        super().__init__(**params)
        self.task = 'regression'
    
    
    def fit(self, X, y):
        ymean, ystd = y.mean(), y.std()
        y = (y - ymean) / (ystd + 1e-5)
        self.Ymean, self.Ystd = ymean, ystd
        super().fit(X, y)
        
    def predict(self, X):
        yp = super().predict(X)
        yp = yp * self.Ystd + self.Ymean
        return yp
    
    
class MLPClassifier(MLPbase):
    
    def __init__(self, **params):
        super().__init__(**params)
        self.task = 'classification'
        
    def predict_proba(self, X):
        yp = self.predict_(X)
        yp = softmax(yp)
        return yp
    
def softmax(x):
    x = x - cp.max(x, axis=1, keepdims=True)
    ex = cp.exp(x)
    s = cp.sum(ex, axis=1, keepdims=True)
    res =  ex/s
    eps = 1e-5
    return cp.clip(res, eps, 1-eps)