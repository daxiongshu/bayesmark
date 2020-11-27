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

    return X.get(), y.get().astype('int32')

def load_data(data_root, name, target, scale_y=False):
    path = f"{data_root}/{name}.csv"
    df = gd.read_csv(path)
    for col in df.columns:
        if col == target and not scale_y:
            continue
        if df[col].dtype == 'O':
            df[col],_ = df[col].factorize()
        mean, std = df[col].mean(), df[col].std()
        df[col] = (df[col] - mean)/(std+1e-5)
        df[col] = df[col].fillna(0)
    return df.drop(target, axis=1).values.get(), df[target].values.get()

def load_california_housing(data_root):
    """https://www.kaggle.com/camnugent/california-housing-prices
    """
    return load_data(data_root, name='housing', target='median_house_value', scale_y=True)

def load_hotel_bookings(data_root):
    """https://www.kaggle.com/jessemostipak/hotel-booking-demand
    """
    return load_data(data_root, name='hotel_bookings', target='is_canceled', scale_y=False)

def load_higgs(data_root):
    """https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
    """
    return load_data(data_root, name='higgs', target='label', scale_y=False)