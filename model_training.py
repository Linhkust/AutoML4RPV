import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from skopt import BayesSearchCV
from skopt.space import Real
import math
from mrmr import mrmr_regression
from category_encoders.target_encoder import TargetEncoder
import time
from scipy.stats import pearsonr
import joblib
from tpot import TPOTRegressor
import h2o
from h2o.automl import H2OAutoML
from pycaret.regression import *
from flaml import AutoML as flaml_AutoML
from supervised.automl import AutoML as mjar_AutoML
from skopt.callbacks import DeltaYStopper
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from benchmark import benchmark_autogluon
import os

# model performance evaluation
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
    train = training_data.reset_index(drop=True)
    test = data.drop(train.index).reset_index(drop=True)

    # Target encoding trained on train data set
    encoder = TargetEncoder().fit(train.iloc[:, :-1], train.iloc[:, -1])

    train_encoded_data = encoder.transform(train.iloc[:, :-1])
    train = pd.concat([train_encoded_data, train.iloc[:, -1]], axis=1)

    test_encoded_data = encoder.transform(test.iloc[:, :-1])
    test = pd.concat([test_encoded_data, test.iloc[:, -1]], axis=1)
    return train, test


# AutoML4PV framework
class AutoML4PV(object):
    # Parameter
    def __init__(self,
                 # saved_path,
                 target,
                 data=None,
                 target_train=None,
                 target_test=None,
                 random_state=0,
                 train_size=0.7):

        # self.path = saved_path
        self.target=target
        self.random_state = random_state
        self.size = train_size

        if data is not None:
            x_train, x_test, y_train, y_test = train_test_split(data.drop(target, axis=1), data[target],
                                                                train_size=train_size,
                                                                random_state=random_state)
            self.train = pd.concat([x_train, y_train], axis=1).reset_index(drop=True)
            self.test = pd.concat([x_test, y_test], axis=1).reset_index(drop=True)

        else:
            self.train = target_train
            self.test = target_test

        scaler = StandardScaler()

        train = self.train.drop(self.target, axis=1)
        test = self.test.drop(self.target, axis=1)

        scaler.fit(train)

        scaled_train = scaler.transform(train)
        scaled_test = scaler.transform(test)

        self.scaled_train = pd.concat([pd.DataFrame(scaled_train, columns=train.columns.tolist()),
                                                    self.train[target]], axis=1)
        self.scaled_test = pd.concat([pd.DataFrame(scaled_test, columns=test.columns.tolist()),
                                      self.test[target]], axis=1)

    # Benchmark models
    def KNN(self):
        parameters = {
            'n_neighbors': list(range(1, 101)),
            'weights': ['uniform', 'distance'],
            }

        model = BayesSearchCV(
            KNeighborsRegressor(),
            parameters,
            cv=5,
            n_iter=50,
            random_state=self.random_state,
            scoring='r2',
            n_jobs=-1)

        early_stopper = DeltaYStopper(delta=0.01, n_best=5)
        model.fit(self.scaled_train.drop(self.target, axis=1), self.scaled_train[self.target], callback=early_stopper)

        cv_accuracy = np.max(model.cv_results_['mean_test_score'])
        return model.best_estimator_, cv_accuracy

    def MLP(self):
        parameters = {'hidden_layer_sizes' : list(range(50, 150)),
        'activation' : ['tanh', 'relu'],
        'alpha' : Real(1e-7, 1e-1),
        'learning_rate_init' : Real(1e-4, 1e-1),
        'learning_rate' : ['constant', 'invscaling', 'adaptive'],
         'max_iter': [500]}

        model = BayesSearchCV(
            MLPRegressor(),
            parameters,
            cv=5,
            n_iter=50,
            random_state=self.random_state,
            scoring='r2',
            n_jobs=-1)

        early_stopper = DeltaYStopper(delta=0.01, n_best=5)
        model.fit(self.scaled_train.drop(self.target, axis=1), self.scaled_train[self.target], callback=early_stopper)

        cv_accuracy = np.max(model.cv_results_['mean_test_score'])
        return model.best_estimator_, cv_accuracy

    def SVR(self):
        parameters = {
        'kernel': ['poly', 'rbf', 'linear', 'sigmoid'],
        'C': Real(1e-4, 25),
        'degree': list(range(1, 5)),
        'max_iter': [3000],
        'tol':[0.001]
        }

        model = BayesSearchCV(
            SVR(),
            parameters,
            cv=5,
            n_iter=50,
            random_state=self.random_state,
            scoring='r2')

        early_stopper = DeltaYStopper(delta=0.01, n_best=5)
        model.fit(self.scaled_train.drop(self.target, axis=1), self.scaled_train[self.target], callback=early_stopper)

        cv_accuracy = np.max(model.cv_results_['mean_test_score'])
        return model.best_estimator_, cv_accuracy

    # Tree-based models
    def RF(self):
        parameters = {'n_estimators': [x for x in range(50, 151)],
                      'max_depth': [x for x in range(10, 101)],
                      'max_features': Real(0.05, 1),
                      'bootstrap': [True, False],
                      'min_samples_split': list(range(2, 21)),
                      'min_samples_leaf': list(range(1, 21))
                      }

        model = BayesSearchCV(
            RandomForestRegressor(),
            parameters,
            cv=5,
            n_iter=50,
            random_state=self.random_state,
            scoring='r2',
            n_jobs=-1)

        early_stopper = DeltaYStopper(delta=0.01, n_best=5)
        model.fit(self.scaled_train.drop(self.target, axis=1), self.scaled_train[self.target], callback=early_stopper)

        cv_accuracy = np.max(model.cv_results_['mean_test_score'])
        return model.best_estimator_, cv_accuracy

    def Extra_Tree(self):
        parameters = {'n_estimators': [x for x in range(50, 151)],
                      'max_depth': [x for x in range(10, 101)],
                      'max_features': Real(0.05, 1),
                      'bootstrap': [True, False],
                      'min_samples_split': list(range(2, 21)),
                      'min_samples_leaf': list(range(1, 21))
                      }

        # model definition
        model = BayesSearchCV(
            ExtraTreesRegressor(),
            parameters,
            cv=5,
            n_iter=50,
            random_state=self.random_state,
            scoring='r2',
            n_jobs=-1)

        early_stopper = DeltaYStopper(delta=0.01, n_best=5)
        model.fit(self.scaled_train.drop(self.target, axis=1), self.scaled_train[self.target], callback=early_stopper)

        cv_accuracy = np.max(model.cv_results_['mean_test_score'])
        return model.best_estimator_, cv_accuracy

    def XGBoost(self):
        parameters = {'n_estimators': [x for x in range(50, 151)],
                      'max_depth': [x for x in range(10, 101)],
                      'learning_rate': Real(1e-3, 1),
                        'subsample': Real(0.5, 1.0),
                        'min_child_weight': list(range(1, 21)),
                        'gamma': Real(1e-4, 20),
                      }

        model = BayesSearchCV(
            XGBRegressor(),
            parameters,
            cv=5,
            n_iter=50,
            random_state=self.random_state,
            scoring='r2',
            n_jobs=-1)

        early_stopper = DeltaYStopper(delta=0.01, n_best=5)
        model.fit(self.scaled_train.drop(self.target, axis=1), self.scaled_train[self.target], callback=early_stopper)

        cv_accuracy = np.max(model.cv_results_['mean_test_score'])
        return model.best_estimator_, cv_accuracy

    def LGBM(self):
        parameters = {'n_estimators': [x for x in range(10, 151)],
                      'max_depth': [x for x in range(10, 101)],
                      'num_leaves': [x for x in range(10, 51)]}

        # model definition
        model = BayesSearchCV(
            LGBMRegressor(verbose=-1),
            parameters,
            cv=5,
            n_iter=50,
            random_state=self.random_state,
            scoring='r2',
            n_jobs=-1)

        early_stopper = DeltaYStopper(delta=0.01, n_best=5)
        model.fit(self.scaled_train.drop(self.target, axis=1), self.scaled_train[self.target], callback=early_stopper)

        cv_accuracy = np.max(model.cv_results_['mean_test_score'])

        return model.best_estimator_, cv_accuracy

    # Ensemble of base learners
    def bagging(self, base_estimator):
        regressor = BaggingRegressor(estimator=base_estimator, n_jobs=-1, random_state=0)
        regressor.fit(self.scaled_train.drop(self.target, axis=1), self.scaled_train[self.target])

        cv_accuracy = np.mean(cross_val_score(regressor,
                                              self.scaled_train.drop(self.target, axis=1),
                                              self.scaled_train[self.target],
                                              scoring='r2',
                                              cv=5))
        return regressor, cv_accuracy

    def two_layer_stacking(self, base_estimators):

        estimator_list = []
        for i, estimator in enumerate(base_estimators):
            estimator_list.append(('Model{}'.format(i), estimator[0]))

        regressor = StackingRegressor(estimators=estimator_list,
                                      final_estimator=LinearRegression(),
                                      n_jobs=-1)

        regressor.fit(self.scaled_train.drop(self.target, axis=1), self.scaled_train[self.target])

        cv_accuracy = np.mean(cross_val_score(regressor,
                                              self.scaled_train.drop(self.target, axis=1),
                                              self.scaled_train[self.target],
                                              scoring='r2',
                                              cv=5))
        return regressor, cv_accuracy

    def weighted_voting(self, base_estimators):
        performances = []
        for i, estimator in enumerate(base_estimators):
            performances.append(estimator[1])

        weights = []
        for performance in performances:
            weights.append(performance / sum(performances))

        estimator_list = []
        for i, estimator in enumerate(base_estimators):
            estimator_list.append(('Model{}'.format(i), estimator[0]))

        regressor = VotingRegressor(estimators=estimator_list,
                                    weights=np.array(weights),
                                    n_jobs=-1)

        regressor.fit(self.scaled_train.drop(self.target, axis=1), self.scaled_train[self.target])

        cv_accuracy = np.mean(cross_val_score(regressor,
                                              self.scaled_train.drop(self.target, axis=1),
                                              self.scaled_train[self.target],
                                              scoring='r2',
                                              cv=5))
        return regressor, cv_accuracy

    def predict(self, estimator):
        y_pred = estimator.predict(self.scaled_test.drop(self.target, axis=1))
        all_features_result = model_performance(self.scaled_test[self.target], y_pred)

        return all_features_result

# detect the columns for target encoding
def target_encoding_detect(df):
    target_encoding_col = []
    for col in df.columns:
        if df[col].dtype == 'object':
            target_encoding_col.append(col)
    return target_encoding_col

# fit all pipelines
def fit(saved_path,
        target,
        data,
        train_size=0.7,
        feature_selection_config='good_quality',
        random_state=0):

    # create results folder
    os.makedirs(saved_path + '/results')

    num_features = len(data.columns) - 1
    results = pd.DataFrame()

    # feature selection
    if feature_selection_config == 'best_quality':
        feature_list = np.linspace(start=1, stop=num_features, num=num_features, dtype=int)
    elif feature_selection_config == 'high_quality':
        feature_list = np.linspace(start=1, stop=num_features, num=int(num_features * 0.6), dtype=int)
    elif feature_selection_config == 'good_quality':
        feature_list = np.linspace(start=1, stop=num_features, num=int(num_features * 0.3), dtype=int)

    target_encoding_col = target_encoding_detect(df=data)

    if len(target_encoding_col) == 0:
        x_train, x_test, y_train, y_test = train_test_split(data.drop(target, axis=1), data[target],
                                                            train_size=train_size,
                                                            random_state=random_state)
        train = pd.concat([x_train, y_train], axis=1).reset_index(drop=True)
        test = pd.concat([x_test, y_test], axis=1).reset_index(drop=True)
    else:
        train, test = reorganize_target(data=data, column=target_encoding_col[0], train_size=train_size, random_state=random_state)

    # Parameters:
    for k in feature_list:
        print(f'Searching using {k} features...')

        X = train.drop(target, axis=1)
        y = train[target]

        selected_features = mrmr_regression(X=X,
                                            y=y,
                                            K=k,
                                            return_scores=False)

        train_test = pd.concat([train, test]).reset_index(drop=True)[selected_features + [target]]
        model = AutoML4PV(target=target, data=train_test, train_size=train_size)

        # start time
        start_time = time.time()

        # 1. base learners
        rt = model.RF()
        joblib.dump(rt[0], saved_path + f'/results/RF_{k}.pkl')
        rt_time = time.time()

        et = model.Extra_Tree()
        joblib.dump(et[0], saved_path + f'/results/ET_{k}.pkl')
        et_time = time.time()

        xg = model.XGBoost()
        joblib.dump(xg[0], saved_path + f'/results/XGBoost_{k}.pkl')
        xg_time = time.time()

        lgbm = model.LGBM()
        joblib.dump(lgbm[0], saved_path + f'/results/LGBM_{k}.pkl')
        lgbm_time = time.time()

        print('Base models are trained successfully!')
        base_estimators = [rt, et, xg, lgbm]

        # Bagging models

        # rt_bagging = model.bagging(rt[0])
        # joblib.dump(rt_bagging[0], './test/results/saved_models/RF_Bagging_{}.pkl'.format(k))
        # rt_bagging_time = time.time()
        #
        # et_bagging = model.bagging(et[0])
        # joblib.dump(et_bagging[0], './test/results/saved_models/ET_Bagging_{}.pkl'.format(k))
        # et_bagging_time = time.time()
        #
        # xg_bagging = model.bagging(xg[0])
        # joblib.dump(xg_bagging[0], './test/results/saved_models/XGBoost_Bagging_{}.pkl'.format(k))
        # xg_bagging_time = time.time()
        #
        # lgbm_bagging = model.bagging(lgbm[0])
        # joblib.dump(lgbm_bagging[0], './test/results/saved_models/LGBM_Bagging_{}.pkl'.format(k))
        # lgbm_bagging_time = time.time()

        # stacking
        base_stacking = model.two_layer_stacking(base_estimators)  # 4B-S
        joblib.dump(base_stacking, saved_path + f'/results/Stacking_{k}.pkl')
        base_stacking_time = time.time()

        # voting
        base_voting = model.weighted_voting(base_estimators)  # 4B-W
        joblib.dump(base_voting, saved_path + f'/results/Voting_{k}.pkl')
        base_voting_time = time.time()

        print('Stacking and Voting models are trained successfully!')
        print('Results collection...')

        # base learner results
        rt_results = model.predict(rt[0])
        et_results = model.predict(et[0])
        xg_results = model.predict(xg[0])
        lgbm_results = model.predict(lgbm[0])

        # ensemble of base learners: results
        # rt_bagging_results = model.predict(rt_bagging[0])
        # et_bagging_results = model.predict(et_bagging[0])
        # xg_bagging_results = model.predict(xg_bagging[0])
        # lgbm_bagging_results = model.predict(lgbm_bagging[0])

        base_stacking_results = model.predict(base_stacking[0])
        base_voting_results = model.predict(base_voting[0])

        et_results['Model'] = 'ET'
        rt_results['Model'] = 'RF'
        xg_results['Model'] = 'XGBoost'
        lgbm_results['Model'] = 'LGBM'
        # rt_bagging_results['Model'] = 'RF_Bagging'
        # et_bagging_results['Model'] = 'ET_Bagging'
        # xg_bagging_results['Model'] = 'XGBoost_Bagging'
        # lgbm_bagging_results['Model'] = 'LGBM_Bagging'
        base_stacking_results['Model'] = 'Stacking'
        base_voting_results['Model'] = 'Voting'

        # combine and identify (number of) selected features
        result = pd.DataFrame([
                               rt_results,
                               et_results,
                               xg_results,
                               lgbm_results,
                               # rt_bagging_results,
                               # et_bagging_results,
                               # xg_bagging_results,
                               # lgbm_bagging_results,
                               base_stacking_results,
                               base_voting_results
                               ])

        training_time = [
                         '%.1f' % (rt_time - start_time),
                         '%.1f' % (et_time - rt_time),
                         '%.1f' % (xg_time - et_time),
                         '%.1f' % (lgbm_time - xg_time), # need revision
                         # '%.1f' % (rt_bagging_time - lgbm_time),
                         # '%.1f' % (et_bagging_time -rt_bagging_time),
                         # '%.1f' % (xg_bagging_time - et_bagging_time),
                         # '%.1f' % (lgbm_bagging_time - xg_bagging_time),
                         '%.1f' % (base_stacking_time - lgbm_time),
                         '%.1f' % (base_voting_time - base_stacking_time),
                         ]

        result['Optimal features'] = str(selected_features)
        result['Number of features'] = len(selected_features)
        result['Training time'] = training_time
        result['Regressor'] = [
                                'RF_{}.pkl'.format(k),
                               'ET_{}.pkl'.format(k),
                               'XGBoost_{}.pkl'.format(k),
                               'LGBM_{}.pkl'.format(k),
                               # 'RF_Bagging_{}.pkl'.format(k),
                               # 'ET_Bagging_{}.pkl'.format(k),
                               # 'XGBoost_Bagging_{}.pkl'.format(k),
                               # 'LGBM_Bagging_{}.pkl'.format(k),
                               'Stacking_{}.pkl'.format(k),
                               'Voting_{}.pkl'.format(k)
                               ]
        print('Results collection completed!')
        results = pd.concat([results, result], axis=0)

    results['Pipeline_ID'] = ['Pipeline_{}'.format(i) for i in range(1, len(results) + 1)]
    col = results.pop('Pipeline_ID')
    results.insert(loc=0, column='Pipeline_ID', value=col)

    results.to_csv(saved_path + '/results/pipeline_results.csv', index=False)
    return results

def benchmark_model(saved_path,
        target,
        data,
        train_size=0.7,
        feature_selection_config='good_quality',
        random_state=0):

    # create results folder
    os.makedirs(saved_path + '/benchmark_results')

    num_features = len(data.columns) - 1
    results = pd.DataFrame()

    # feature selection
    if feature_selection_config == 'best_quality':
        feature_list = np.linspace(start=1, stop=num_features, num=num_features, dtype=int)
    elif feature_selection_config == 'high_quality':
        feature_list = np.linspace(start=1, stop=num_features, num=int(num_features * 0.6), dtype=int)
    elif feature_selection_config == 'good_quality':
        feature_list = np.linspace(start=1, stop=num_features, num=int(num_features * 0.3), dtype=int)

    target_encoding_col = target_encoding_detect(df=data)

    if len(target_encoding_col) == 0:
        x_train, x_test, y_train, y_test = train_test_split(data.drop(target, axis=1), data[target],
                                                            train_size=train_size,
                                                            random_state=random_state)
        train = pd.concat([x_train, y_train], axis=1).reset_index(drop=True)
        test = pd.concat([x_test, y_test], axis=1).reset_index(drop=True)
    else:
        train, test = reorganize_target(data=data, column=target_encoding_col[0], train_size=train_size, random_state=random_state)

    # Parameters:
    for k in feature_list:
        print(f'Searching using {k} features...')
        X = train.drop(target, axis=1)
        y = train[target]

        selected_features = mrmr_regression(X=X,
                                            y=y,
                                            K=k,
                                            return_scores=False)

        train_test = pd.concat([train, test]).reset_index(drop=True)[selected_features + [target]]
        model = AutoML4PV(target=target, data=train_test , train_size=train_size)

        # start time
        start_time = time.time()

        # 1. base learners
        knn = model.KNN()
        joblib.dump(knn[0], saved_path + f'/benchmark_results/KNN_{k}.pkl')
        knn_time = time.time()

        ann = model.MLP()
        joblib.dump(ann[0], saved_path + f'/benchmark_results/ANN_{k}.pkl')
        ann_time = time.time()

        svr = model.SVR()
        joblib.dump(svr[0], saved_path + f'/benchmark_results/SVR_{k}.pkl')
        svr_time = time.time()

        print('Base benchmark models are trained successfully!')
        print('Results collection...')

        # base learner results
        knn_results = model.predict(knn[0])
        ann_results = model.predict(ann[0])
        svr_results = model.predict(svr[0])

        knn_results['Model'] = 'KNN'
        ann_results['Model'] = 'ANN'
        svr_results['Model'] = 'SVR'

        # combine and identify (number of) selected features
        result = pd.DataFrame([
            knn_results,
            ann_results,
            svr_results
        ])

        training_time = [
            '%.1f' % (knn_time - start_time),
            '%.1f' % (ann_time - knn_time),
            '%.1f' % (svr_time - ann_time)]

        result['Optimal features'] = str(selected_features)
        result['Number of features'] = len(selected_features)
        result['Training time'] = training_time
        result['Regressor'] = [
            'KNN_{}.pkl'.format(k),
            'ANN_{}.pkl'.format(k),
            'SVR_{}.pkl'.format(k)]
        print('Results collection completed!')
        results = pd.concat([results, result], axis=0)

    results['Pipeline_ID'] = ['Pipeline_{}'.format(i) for i in range(1, len(results) + 1)]
    col = results.pop('Pipeline_ID')
    results.insert(loc=0, column='Pipeline_ID', value=col)

    results.to_csv(saved_path + '/benchmark_results/pipeline_results.csv', index=False)
    return results

'''
Benchmark: TPOT, H2O, FLAML, mjar-supervised, PyCaret, AutoGluon
'''
def benchmark_tpot(train,
                   test,
                   target,
                   generation=5,
                   population_size=10,
                   mutation=0.7,
                   crossover=0.2,
                   verbosity=2):

    tpot = TPOTRegressor(generations=generation,
                         population_size=population_size,
                         mutation_rate=mutation,
                         crossover_rate=crossover,
                         verbosity=verbosity)

    start = time.time()
    tpot.fit(train.drop(target, axis=1), train[target])
    finish = time.time()

    running_time = ("%.2f" % ((finish - start) / 60))
    y_pred = tpot.predict(test.drop(target, axis=1))

    performance = model_performance(test[target], y_pred)
    performance['Framework'] = 'TPOT'
    performance['Training time'] = running_time
    # tpot.export(output_file_name='tpot_deployment.py', data_file_path=saved_path)
    return performance

# Using H2O AutoML framework to derive the optimal ML configuration
def benchmark_h2o(train, test,target,
                max_models=10):

    h2o.init()
    train_frame = h2o.H2OFrame(train)
    test_frame = h2o.H2OFrame(test)

    # Run AutoML for 10 base models
    aml = H2OAutoML(max_models=max_models)

    start = time.time()
    aml.train(x=train.drop(target, axis=1).columns.tolist(), y=target, training_frame=train_frame)
    finish = time.time()
    running_time = ("%.2f" % ((finish - start) / 60))

    y_pred = aml.leader.predict(test_frame).as_data_frame().loc[:, 'predict']
    y_test = test[target]

    performance = model_performance(y_test, y_pred)
    performance['Framework'] = 'H2O'
    performance['Training time'] = running_time

    # h2o leanderboard
    # lb = h2o.automl.get_leaderboard(aml, extra_columns = "ALL")
    return performance


# Using FLAML to train ML models
def benchmark_flaml(train, test, target,
                time_budget=60):

    automl = flaml_AutoML()
    start = time.time()
    automl.fit(X_train=train.drop(target, axis=1),
               y_train=train[target],
               task='regression',
               time_budget=time_budget)
    finish = time.time()
    running_time = ("%.2f" % ((finish - start) / 60)) # minutes

    # Print the best model
    model = automl.model.estimator
    y_pred = model.predict(test.drop(target, axis=1))
    performance = model_performance(test[target].values, y_pred)
    performance['Framework'] = 'FLAML'
    performance['Training time'] = running_time
    return performance

# Using mjar to train ML models
def benchmark_mjar(train, test, target,
                total_time_limit=3600,
                mode='Explain'):

    automl = mjar_AutoML(total_time_limit=total_time_limit, mode=mode)

    start = time.time()
    automl.fit(train.drop(target, axis=1), train[target])
    finish = time.time()
    running_time = ("%.2f" % ((finish - start) / 60))

    y_pred = automl.predict(test.drop(target, axis=1))

    performance = model_performance(test[target].values, y_pred)
    performance['Framework'] = 'mjar_supervised'
    performance['Training time'] = running_time
    return performance

# Using PyCaret to train ML models
def benchmark_pycaret(train,
                      test,
                      target,
                      feature_selection_method='classic',
                      n_features_to_select=0.2
                      ):

    start = time.time()
    exp = setup(train, index=False,
                target=target,
                test_data=test,
                feature_selection=True,
                feature_selection_method=feature_selection_method,
                n_features_to_select=n_features_to_select)

    best = compare_models()
    finish= time.time()
    running_time = ("%.2f" % ((finish - start) / 60))

    y_pred = predict_model(best, data=test).prediction_label
    performance = model_performance(test[target].values, y_pred.values)
    performance['Framework'] = 'PyCaret'
    performance['Training time'] = running_time
    return performance

# Benchmark AutoML frameworks with specified number of selected features (k)
def benchmark(target,
              data,
              k,
              automl,
              train_size=0.7):

    target_encoding_col = target_encoding_detect(df=data)

    if len(target_encoding_col) == 0:
        base0 = AutoML4PV(target=target, data=data, train_size=train_size)
    else:
        train, test = reorganize_target(data=data, column=target_encoding_col[0], train_size=train_size, random_state=0)
        base0 = AutoML4PV(target=target, target_train=train, target_test=test, train_size=train_size)

    print(f'Searching using {k} features...')

    X = base0.scaled_train.drop(target, axis=1)
    y = base0.scaled_train[target]
    selected_features = mrmr_regression(X=X,
                                        y=y,
                                        K=k,
                                        return_scores=False)
    if automl == 'TPOT':
        return benchmark_tpot(base0.scaled_train[selected_features + [target]],
                              base0.scaled_test[selected_features + [target]],
                              target)

    elif automl == 'H2O':
        return benchmark_h2o(base0.scaled_train[selected_features + [target]],
                              base0.scaled_test[selected_features + [target]],
                              target)

    elif automl == 'FLAML':
        return benchmark_flaml(base0.scaled_train[selected_features + [target]],
                             base0.scaled_test[selected_features + [target]],
                             target)

    elif automl == 'mjar-supervised':
        return benchmark_mjar(base0.scaled_train[selected_features + [target]],
                             base0.scaled_test[selected_features + [target]],
                             target)

    elif automl == 'PyCaret':
        return benchmark_pycaret(base0.scaled_train[selected_features + [target]],
                             base0.scaled_test[selected_features + [target]],
                             target)

    elif automl == 'AutoGluon':
        return benchmark_autogluon(base0.scaled_train[selected_features + [target]],
                             base0.scaled_test[selected_features + [target]],
                             target)

# Benchmark AutoML frameworks with specified feature selection configurations
# (best quality, high quality, and good quality)
def automl_benchmark(data,
                     feature_selection_config,
                     target,
                     train_size,
                     automl,
                     saved_path,
                     random_state=0):
    num_features = len(data.columns) - 1

    # feature selection
    if feature_selection_config == 'best_quality':
        feature_list = np.linspace(start=1, stop=num_features, num=num_features, dtype=int)
    elif feature_selection_config == 'high_quality':
        feature_list = np.linspace(start=1, stop=num_features, num=int(num_features * 0.6), dtype=int)
    elif feature_selection_config == 'good_quality':
        feature_list = np.linspace(start=1, stop=num_features, num=int(num_features * 0.3), dtype=int)

    target_encoding_col = target_encoding_detect(df=data)

    if len(target_encoding_col) == 0:
        x_train, x_test, y_train, y_test = train_test_split(data.drop(target, axis=1), data[target],
                                                            train_size=train_size,
                                                            random_state=random_state)
        train = pd.concat([x_train, y_train], axis=1).reset_index(drop=True)
        test = pd.concat([x_test, y_test], axis=1).reset_index(drop=True)
    else:
        train, test = reorganize_target(data=data, column=target_encoding_col[0], train_size=train_size,
                                        random_state=random_state)

    results = pd.DataFrame()
    # Parameters:
    for k in feature_list:
        print(f'Searching using {k} features with {automl}...')

        X = train.drop(target, axis=1)
        y = train[target]

        selected_features = mrmr_regression(X=X,
                                            y=y,
                                            K=k,
                                            return_scores=False)

        train_test = pd.concat([train, test]).reset_index(drop=True)[selected_features + [target]]

        start = time.time()
        result = benchmark(target=target,
                  data=train_test,
                  k=k,
                  automl=automl,
                  train_size=train_size)
        finish = time.time()

        result['Optimal features'] = str(selected_features)
        result['Number of features'] = len(selected_features)
        result['Training time'] = '%.1f' % (finish - start)
        result['AutoML_Features'] = ['{}_{}'.format(automl, k)]

        result = pd.DataFrame(result)
        results = pd.concat([results, result], axis=0)

    results['Pipeline_ID'] = ['Pipeline_{}'.format(i) for i in range(1, len(results) + 1)]
    col = results.pop('Pipeline_ID')
    results.insert(loc=0, column='Pipeline_ID', value=col)

    results.to_csv(saved_path + f'/{automl}_pipeline_results.csv', index=False)
    return results


def main():
    ''' TPOT H2O FLAML mjar-supervised PyCaret AutoGluon Geochemistrypi'''
    region = 'sg'

    ml_list = ['TPOT', 'H2O', 'FLAML', 'mjar-supervised', 'PyCaret']
    automl_models = ml_list

    results = []
    for automl_model in automl_models:
        print('[' + automl_model + ']')
        result = benchmark(target='Transacted Price ($)',
                  data=pd.read_csv(f'./paper_test/{region}_test/train.csv'),
                  # data=pd.read_csv(f'./paper_test/taipei_test/taipei_test.csv'),
                  k=25,
                  automl=automl_model,
                  train_size=0.7)
        results.append(result)
    results = pd.DataFrame(results)
    results.to_csv(f'./paper_test/{region}_test/benchmark_all.csv')
    # results.to_csv(f'./paper_test/taipei_test/benchmark_all.csv')

def autogluon_():
    region = 'sg'
    result = benchmark(
                       target='Transacted Price ($)',
                       data=pd.read_csv(f'./paper_test/{region}_test/train.csv'),
                       k=6,
                       automl='AutoGluon',
                       train_size=0.7)
    result = pd.DataFrame(result, index=[0])
    result.to_csv(f'./paper_test/{region}_test/autogluon_benchmark.csv')

def autogluon_taipei():
    result = benchmark(
                       target='Y house price of unit area',
                       data=pd.read_csv(f'paper_test/taipei_test/taipei.csv'),
                       k=6,
                       automl='AutoGluon',
                       train_size=0.7)
    result = pd.DataFrame(result, index=[0])
    result.to_csv(f'./paper_test/taipei/autogluon_benchmark.csv')

if __name__ == "__main__":
    ''' '''
    automl_benchmark(data=pd.read_csv(f'./paper_test/taipei_test/taipei.csv'),
                     feature_selection_config='good_quality',
                     target='Y house price of unit area',
                     train_size=0.7,
                     automl='FLAML',
                     saved_path='./paper_test/taipei_test',
                     random_state=0)

    ''' TPOT H2O FLAML mjar-supervised PyCaret AutoGluon'''
    #main()
    # autogluon_taipei()

    ''' RF, EXTRA TREE, XGBOOST, LGBM, STACKING, VOTING'''
    # Real estate valuation dataset
    # fit(target='Y house price of unit area',
    #     data=pd.read_csv(f'./paper_test/taipei_test/taipei_test.csv'),
    #     saved_path='./paper_test/taipei_test',
    #     train_size=0.7,
    #     feature_selection_config='best_quality')

    # fit(target='Price',
    #     data=pd.read_csv('./paper_test/lo_test/train.csv'),
    #     saved_path='./paper_test/lo_test',
    #     train_size=0.7,
    #     feature_selection_config='good_quality')

    # fit(target='SALE PRICE',
    #     data=pd.read_csv('./paper_test/ny_test/train.csv'),
    #     saved_path='./paper_test/ny_test',
    #     train_size=0.7,
    #     feature_selection_config='good_quality')

    # fit(target='Transacted Price ($)',
    #                 data=pd.read_csv('./paper_test/sg_test/train.csv'),
    #                 saved_path='./paper_test/sg_test',
    #                 train_size=0.7,
    #                 feature_selection_config='good_quality')

    ''' SVR, KNN, ANN'''
    # benchmark_model(target='Price',
    #     data=pd.read_csv('./paper_test/lo_test/train.csv'),
    #     saved_path='./paper_test/lo_test',
    #     train_size=0.7,
    #     feature_selection_config='good_quality')
    #
    # benchmark_model(target='SALE PRICE',
    #     data=pd.read_csv('./paper_test/ny_test/train.csv'),
    #     saved_path='./paper_test/ny_test',
    #     train_size=0.7,
    #     feature_selection_config='good_quality')
    #
    # benchmark_model(target='Transacted Price ($)',
    #     data=pd.read_csv('./paper_test/sg_test/train.csv'),
    #     saved_path='./paper_test/sg_test',
    #     train_size=0.7,
    #     feature_selection_config='good_quality')