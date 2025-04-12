import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -123525544703.38412
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.01, max_depth=1, min_child_weight=12, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.4, verbosity=0)),
    SelectFwe(score_func=f_regression, alpha=0.026000000000000002),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=7, min_samples_leaf=16, min_samples_split=5)),
    RandomForestRegressor(bootstrap=False, max_features=0.7500000000000001, min_samples_leaf=11, min_samples_split=14, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
