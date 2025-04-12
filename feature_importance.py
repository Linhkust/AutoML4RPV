import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from matplotlib import pyplot as plt
from alepython import ale_plot
import shap
import plotly.express as px
import joblib

class feature_importance_(object):
    # Parameter
    def __init__(self,
                 x,
                 y,
                 model,
                 saved_path):
        self.x = x
        self.y = y
        self.model = model
        self.path = saved_path
        # self.random_state = random_state
        # self.size = train_size
        # self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y,
        #                                                                         train_size=train_size,
        #                                                                         random_state=random_state)

    def pdp(self, feature_name):
        self.model.fit(self.x, self.y)
        feature_names = self.x.columns
        feature_index = 0
        for index, feature in enumerate(feature_names):
            if feature == feature_name:
                feature_index = index
                break
        fig, ax = plt.subplots(1, 1)
        PartialDependenceDisplay.from_estimator(self.model, self.x,
                                                features=[feature_index],
                                                kind='average',
                                                ax=ax,
                                                feature_names=feature_names,
                                                n_jobs=-1)
        return fig

    # permutation importance analysis
    def pfi(self):
        self.model.fit(self.x, self.y)
        result = permutation_importance(self.model, self.x, self.y,
                                        scoring='neg_mean_absolute_percentage_error',
                                        n_repeats=10)

        sorted_importance_idx = result.importances_mean.argsort()
        importance = pd.DataFrame(
            result.importances[sorted_importance_idx].T,
            columns=self.x.columns[sorted_importance_idx], )
        importance.to_csv(self.path + '/pfi.csv', index=False)
        return importance

    def pfi_figure(self):
        importance = pd.read_csv(self.path + '/pfi.csv')
        # importance = self.pfi()
        ax = importance.plot.box(vert=False, whis=10, figsize=(8, 7))
        ax.set_title("Permutation Importance")
        ax.axvline(x=0, color="k", linestyle="--")
        ax.set_xlabel("Decrease in accuracy score")
        ax.figure.tight_layout()
        return ax

    # ALE plot
    def ale(self, feature_name):
        self.model.fit(self.x, self.y)
        ale_plot(self.model, self.x, features=feature_name, bins=10, monte_carlo=True)

    # SHAP feature importance and dependence plot
    def shap_analysis(self):
        feature_names = self.x.columns
        self.model.fit(self.x, self.y)

        # explain process
        explainer = shap.Explainer(self.model.predict, self.x)
        shap_values = explainer(self.x)

        shap_values_result = pd.DataFrame(shap_values.values, columns=feature_names)
        data_values_result = pd.DataFrame(shap_values.data, columns=feature_names)

        shap_values_result.to_csv(self.path + '/shap_values.csv', index=False)
        data_values_result.to_csv(self.path + '/data_values.csv', index=False)

        return shap_values_result, data_values_result

    def shap_summary(self, max_display):
        # shap_values = self.shap_analysis()[0]
        shap_values = pd.read_csv(self.path + '/shap_values.csv')
        feature_names = shap_values.columns
        fig = shap.summary_plot(shap_values=shap_values,
                                max_display=max_display,
                                feature_names=feature_names,
                                plot_type="bar",
                                plot_size=[10, 7], show=False)
        return fig

    def shap_dependence(self, feature_name, interactive_feature=None):
        # shap_values, data_values = self.shap_analysis()
        shap_values = pd.read_csv(self.path + '/shap_values.csv')
        data_values = pd.read_csv(self.path + '/data_values.csv')

        feature_names = shap_values.columns

        fig, ax = plt.subplots(figsize=(6, 6))
        shap.dependence_plot(ind=feature_name,
                             shap_values=shap_values.values,
                             features=data_values,
                             feature_names=feature_names,
                             interaction_index=interactive_feature,
                             alpha=0.7,
                             show=False,
                             ax=ax)
        return fig

    def shap_spatial(self, selected_feature):

        data_values = pd.read_csv(self.path + '/data_values.csv')
        shap_values = pd.read_csv(self.path + '/shap_values.csv')

        # change column name
        data_values.columns = [name+'_data' for name in data_values.columns.tolist()]
        shap_values.columns = [name + '_shap' for name in shap_values.columns.tolist()]
        values = pd.concat([data_values, shap_values], axis=1)

        # values = pd.concat([values, pd.DataFrame(self.y_test).reset_index()], axis=1)
        fig = px.scatter_mapbox(values,
                                lat='Latitude_data',
                                lon='Longitude_data',
                                # size=values.columns.tolist()[-1],
                                size=selected_feature+'_data',
                                color=selected_feature+'_shap',
                                center=dict(lat=values['Latitude_data'].mean(), lon=values['Longitude_data'].mean()),
                                mapbox_style='carto-positron',
                                color_continuous_scale=px.colors.sequential.Viridis,
                                opacity=0.8,
                                zoom=11)
        return fig


if __name__ == "__main__":
    data = pd.read_csv('./test/train.csv')
    # model = joblib.load('./test/results/saved_models/{}'.format(best_pipeline.loc[0, 'Regressor']))
    # fi = feature_importance_(x=data.loc[:, eval(best_pipeline.loc[0, 'Optimal features'])],
    #                          y=data.iloc[:, -1],
    #                          model=model[0])
    # fi.shap_spatial(selected_feature='Area (SQFT)')
    # fi.pdp(feature_name='Area (SQFT)')
    # fi.ale_analysis(feature_name='Floor Level encoding')
    # fi.shap_summary(max_display=len(data.iloc[:, :-1].columns))
    # fi.shap_dependence(feature_name='Area (SQFT)')
    # selected_features = mrmr_regression(X=data.iloc[:, :-1], y=data.iloc[:, -1], K=37)
    # fi = importance(x=data.loc[:, selected_features], y=data.iloc[:, -1], region='singapore', model=model)
