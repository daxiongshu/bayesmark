import cudf as gd
import cupy as cp
import os

def load_mnist(data_root):
    path = f"{data_root}/mnist_train.csv"
    if os.path.exists(path) == False:
        os.system(f"wget https://pjreddie.com/media/files/mnist_train.csv {path}")
    df = gd.read_csv(path, header=None)
    df.columns = ['target'] + [f'p_{i}' for i in range(df.shape[1]-1)]

    X = df.drop('target', axis=1).values/255
    y = df['target'].values

    return X.get(), y.get()

def load_california_housing(data_root):
    path = f"{data_root}/"
    