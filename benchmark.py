from operator import index
import numpy as np
# from autogluon.tabular import TabularDataset, TabularPredictor
# from autogluon.features.generators import AutoMLPipelineFeatureGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost.dask import train
# from pycaret.regression import *
from sklearn.preprocessing import StandardScaler
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import math
from category_encoders import TargetEncoder
from scipy.stats import pearsonr

def model_performance(y_test, y_pred):
    # ML perspective
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Property valuation perspective
    ratio = y_pred / y_test
    ratio_median = np.median(ratio)
    cod = np.sum(abs(ratio - ratio_median)) / (len(ratio) * ratio_median) * 100

    # Model performance results
    # Model performance results
    performance_result = {'RMSE': "%.4f" % math.sqrt(mse),
                          '%RMSE': "%.4f" % (((math.sqrt(mse)) / abs(np.mean(y_test))) * 100) + '%',
                          'MAE': "%.4f" % mae,
                          '%MAE': "%.4f" % ((mae / abs(np.mean(y_test))) * 100) + '%',
                          'MAPE': "%.4f" % (mean_absolute_percentage_error(y_pred, y_test) * 100) + '%',
                          'R2': "%.4f" % r2,
                          'COD': "%.4f" % cod,
                          'std': y_pred.std(),
                          'rho': pearsonr(y_pred, y_test)[0],
                          'ref': y_test.std()
                          }

    return performance_result

# target encoding
def reorganize_target(data, column, train_size=0.7, random_state=0):
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    training_data_samples = int(len(x) * train_size)
    training_data = pd.DataFrame()

    single_value_category = 0

    for category in data[column].unique():
        category_info = data[data[column] == category]
        random_extraction = category_info.sample(n=int(
            (training_data_samples - single_value_category) * (
                    len(category_info) / (len(data) - single_value_category))),
            random_state=random_state)
        training_data = pd.concat([training_data, random_extraction])

    # train and test data
    train = training_data.reset_index().drop('index', axis=1)
    test = data.drop(train.index).reset_index().drop('index', axis=1)

    # Target encoding trained on train data set
    encoder = TargetEncoder().fit(train.iloc[:, :-1], train.iloc[:, -1])

    train_encoded_data = encoder.transform(train.iloc[:, :-1])
    train = pd.concat([train_encoded_data, train.iloc[:, -1]], axis=1)

    test_encoded_data = encoder.transform(test.iloc[:, :-1])
    test = pd.concat([test_encoded_data, test.iloc[:, -1]], axis=1)
    return train, test

def train_test_set(data, train_size=0.7, random_state=0):
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        train_size=train_size,
                                                        random_state=random_state)
    return x_train, x_test, y_train, y_test

def autogluon_benchmark(target, train_path, random_state=0):

    data = TabularDataset(train_path)
    x_train, x_test, y_train, y_test = train_test_set(data, random_state=random_state)
    train =  pd.concat([x_train, y_train], axis=1)
    test = pd.concat([x_test, y_test], axis=1)

    # train_feature_generator = AutoMLPipelineFeatureGenerator()

    # # train fit
    # train_data = train_feature_generator.fit_transform(X=train.iloc[:, :-1])
    # train_data = pd.concat([train_data, train.iloc[:, -1]], axis=1)

    # # test fit
    # test_data = train_feature_generator.transform(X=test.iloc[:, :-1])
    # test_data = pd.concat([test_data, test.iloc[:, -1]], axis=1)

    # predictor = TabularPredictor(label=target).fit(train_data, time_limit=300)
    # print(predictor.evaluate(test_data))

    predictor = TabularPredictor(label=target).fit(train, time_limit=300)
    print(predictor.evaluate(test))

def pycaret_benchmark(target, train_path, random_state=0):

    data = pd.read_csv(train_path)
    x_train, x_test, y_train, y_test = train_test_set(data, random_state=random_state)
    train = pd.concat([x_train, y_train], axis=1)
    test = pd.concat([x_test, y_test], axis=1)

    exp_name = setup(data=train,
                     target=target,
                     test_data=test,
                     session_id=0, index=False)

    best_model = compare_models()
    print(best_model)

def benchmark_autogluon(train, test, target, experiment_id=0):
    start = time.time()

    predictor = TabularPredictor(label=target).fit(train)
    finish = time.time()
    running_time = ("%.2f" % ((finish - start) / 60))

    y_pred = predictor.predict(test.drop(target, axis=1))

    performance = model_performance(test[target].values, y_pred)
    performance['Framework'] = 'AutoGluon'
    performance['Training time'] = running_time
    performance['Experiment_ID'] = experiment_id

    return performance

'''

Use PyCaret and AutoGluon for raw data processing and model training

'''
def pycaret_regions(region, regions, target):
    pycaret_benchmark(target=target,
                      train_path=f'./paper_test/{region}_test/{regions}.csv')


def autogluon_regions(region, regions, target):
    autogluon_benchmark(target=target,
                        train_path=f'./paper_test/{region}_test/{regions}.csv')

if __name__ == "__main__":
    autogluon_regions(region='ny', regions='newyork', target='SALE PRICE')
    # autogluon_regions(region='lo', regions='london', target='Price')
    # autogluon_regions(region='sg', regions='singapore', target='Transacted Price ($)')
    # pycaret_regions(region='ny', regions='newyork', target='SALE PRICE')
    # pycaret_regions(region='lo', regions='london', target='Price')
    # pycaret_regions(region='sg', regions='singapore', target='Transacted Price ($)')