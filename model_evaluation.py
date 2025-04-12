import os
import numpy as np
import pandas as pd
from autorank import autorank, plot_stats
from scipy.stats import wilcoxon
import matplotlib.patches as patches
import warnings
from td.taylorDiagram import TaylorDiagram
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns
from sklearn.metrics import r2_score
import scikit_posthocs as sp

warnings.filterwarnings('ignore')

def mrmr_performance(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    merged_data = pd.DataFrame()

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)
        merged_data = pd.concat([merged_data, data], ignore_index=True)

    merged_data = merged_data.sort_values('Number of features')
    merged_data.to_csv('results.csv', index=False)

def mrmr_performance_plot(data, metrics):
    fig, ax = plt.subplots(figsize=(7, 5))
    label = ['Random Forest', 'Extra Trees', 'XGBoost', 'LightGBM', 'Stacking', 'Voting']
    for index, model in enumerate(['RF', 'ET', 'XGBoost', 'LGBM', 'Stacking', 'Voting']):
        model_data = data[data['Model'] == model]
        ax.plot(model_data['Number of features'], model_data[metrics], label=label[index])

    ax.set_title('Model performances with MRMR')
    ax.set_xlabel('Number of selected features')
    ax.set_ylabel('{} (test set)'.format(metrics))
    ax.legend()
    plt.show()


def wilcoxon_paired_p_value(data, model1, model2, indicator):

    df = data[['RMSE', '%RMSE', 'MAE', 'MAPE', 'R2', 'COD', 'Model']]
    df['%RMSE'] = df['%RMSE'].str.replace('%', '').astype(float)
    # data['%MAE'] = data['%MAE'].str.replace('%', '').astype(float)
    df['MAPE'] = df['MAPE'].str.replace('%', '').astype(float)

    d = df[df['Model'] == model1][indicator].reset_index().drop('index', axis=1) - df[df['Model'] == model2][
        indicator].reset_index().drop('index', axis=1)
    if indicator == 'R2':
        res = wilcoxon(d, alternative='greater')
    else:
        res = wilcoxon(d, alternative='less')
    return res.pvalue[0]


def wilcoxon_paired_matrix_plot(region):

    automl4rpv_data = pd.read_csv(f'./paper_test/{region}_test/results/pipeline_results.csv')
    ml_data = pd.read_csv(f'./paper_test/{region}_test/benchmark_results/pipeline_results.csv')
    data = pd.concat([automl4rpv_data, ml_data])

    fig, ax = plt.subplots(2, 3, figsize=(22, 10))

    indicators = ['RMSE', '%RMSE', 'MAE', 'MAPE', 'R2', 'COD']
    new_indicators = ['RMSE', 'NRMSE', 'MAE', 'MAPE', 'R2', 'COD']
    for m in range(2):
        for n in range(3):
            p_values = np.zeros(shape=(9, 9))
            for i, model1 in enumerate(['SVR','KNN', 'ANN',
                                        'RF',
                                        'ET',
                                        'XGBoost',
                                        'LGBM',
                                        'Voting',
                                        'Stacking'
                                        ]):
                for j, model2 in enumerate(['SVR','KNN', 'ANN',
                                             'RF',
                                            'ET',
                                            'XGBoost',
                                            'LGBM',
                                            'Voting',
                                            'Stacking'
                                            ]):
                    p_values[i, j] = 1 if model1 == model2 else wilcoxon_paired_p_value(data,
                                                                                        model1, model2,
                                                                                        indicator=indicators[3 * m + n])

            for i in range(p_values.shape[0]):
                for j in range(p_values.shape[1]):
                    color = 'green' if p_values[i, j] < 0.1 else 'red'
                    radius = 0.15

                    circle = patches.Circle((j, i), radius=radius, facecolor=color, alpha=0.6)
                    circle1 = patches.Circle((j, j), radius=radius, facecolor='gray')
                    ax[m, n].add_patch(circle)
                    ax[m, n].add_patch(circle1)

            # 设置坐标轴范围和标签
            ax[m, n].set_xticks(ticks=np.arange(p_values.shape[1]), labels=['SVR','KNN', 'ANN','RF',
                                                                            'ET',
                                                                            'XGBoost',
                                                                            'LGBM',
                                                                            'Voting',
                                                                            'Stacking'
                                                                            ])
            ax[m, n].set_yticks(ticks=np.arange(p_values.shape[0]), labels=['SVR','KNN', 'ANN','RF',
                                                                            'ET',
                                                                            'XGBoost',
                                                                            'LGBM',
                                                                            'Voting',
                                                                            'Stacking'
                                                                            ])
            ax[m, n].set_xlim([-0.5, p_values.shape[1] - 0.5])
            ax[m, n].set_ylim([-0.5, p_values.shape[0] - 0.5])
            ax[m, n].set_title(new_indicators[3 * m + n])
            ax[m, n].grid(axis='y')
    plt.savefig(f'./paper_test/{region}_test/wilcoxon_paired_matrix_plot.png', format='PNG', dpi=500,
                bbox_inches='tight')
    # plt.show()

def wilcoxon_test(region, metric):
    automl4rpv_data = pd.read_csv(f'./paper_test/{region}_test/results/pipeline_results.csv')
    ml_data = pd.read_csv(f'./paper_test/{region}_test/benchmark_results/pipeline_results.csv')
    data = pd.concat([automl4rpv_data, ml_data])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    p_values = np.zeros(shape=(9, 9))
    for i, model1 in enumerate(['SVR', 'KNN', 'ANN',
                                'RF',
                                'ET',
                                'XGBoost',
                                'LGBM',
                                'Voting',
                                'Stacking'
                                ]):
        for j, model2 in enumerate(['SVR', 'KNN', 'ANN',
                                    'RF',
                                    'ET',
                                    'XGBoost',
                                    'LGBM',
                                    'Voting',
                                    'Stacking'
                                    ]):
            p_values[i, j] = 1 if model1 == model2 else wilcoxon_paired_p_value(data,
                                                                                model1, model2,
                                                                                indicator=metric)

    for i in range(p_values.shape[0]):
        for j in range(p_values.shape[1]):
            color = 'green' if p_values[i, j] < 0.1 else 'red'
            radius = 0.15

            circle = patches.Circle((j, i), radius=radius, facecolor=color, alpha=0.6)
            circle1 = patches.Circle((j, j), radius=radius, facecolor='gray')
            ax.add_patch(circle)
            ax.add_patch(circle1)

    # 设置坐标轴范围和标签
    ax.set_xticks(ticks=np.arange(p_values.shape[1]), labels=['SVR', 'KNN', 'ANN', 'RF',
                                                                    'ET',
                                                                    'XGBoost',
                                                                    'LGBM',
                                                                    'Voting',
                                                                    'Stacking'
                                                                    ])
    ax.set_yticks(ticks=np.arange(p_values.shape[0]), labels=['SVR', 'KNN', 'ANN', 'RF',
                                                                    'ET',
                                                                    'XGBoost',
                                                                    'LGBM',
                                                                    'Voting',
                                                                    'Stacking'
                                                                    ])
    ax.set_xlim([-0.5, p_values.shape[1] - 0.5])
    ax.set_ylim([-0.5, p_values.shape[0] - 0.5])

    ax.set_title(['(a) New York', '(b) London', '(c) Singapore'][['ny', 'lo', 'sg'].index(region)])
    ax.grid(axis='y')
    plt.savefig(f'./paper_test/{region}_test/wilcoxon_paired_matrix_plot.png', format='PNG', dpi=500,
                bbox_inches='tight')
    # plt.show()

# Model performance Leaderboard
def pipelines(data):
    performance_data = data[['RMSE', '%RMSE', 'MAE', 'MAPE', 'R2', 'COD']]
    performance_data['%RMSE'] = data['%RMSE'].str.replace('%', '').astype(float)
    performance_data['MAPE'] = data['MAPE'].str.replace('%', '').astype(float)
    performance_data['R2'] = data.apply(lambda x: 1 - x['R2'], axis=1)

    performance_data_ranking = performance_data.rank(method='min')
    performance_data['mean_ranks'] = performance_data_ranking.mean(axis=1)

    return performance_data

def taylor_diagram(saved_path,
                   data,
                   pipeline_ids):

    df = pipelines(data)

    # Reference std
    stdref = df.loc[0, 'ref']

    # Samples std,rho,name
    samples = df.loc[:, ['std', 'rho', 'Pipeline_ID']]
    samples = samples[samples['Pipeline_ID'].isin(pipeline_ids)]

    samples = samples.values.tolist()

    fig = plt.figure(figsize=(10, 10))
    dia = TaylorDiagram(stdref, fig=fig, label='Reference', extend=False)
    dia.samplePoints[0].set_color('r')  # Mark reference point as a red star

    # color label
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Add models to Taylor diagram
    for i, (stddev, corrcoef, name) in enumerate(samples):
        dia.add_sample(stddev, corrcoef,
                       marker='o', ms=6, ls='', alpha=0.5,
                       mfc=colors[i],
                       label=name)

    # Add RMS contours, and label them
    contours = dia.add_contours(levels=5, colors='0.5')  # 5 levels in grey
    plt.clabel(contours, inline=1, fontsize=10, fmt='%.0f')

    dia.add_grid()  # Add grid
    dia._ax.axis[:].major_ticks.set_tick_out(True)  # Put ticks outward

    # Add a figure legend and title
    fig.legend(dia.samplePoints,
               [p.get_label() for p in dia.samplePoints],
               numpoints=1, prop=dict(size='small'), loc='upper right')

    fig.suptitle("Taylor diagram", size='x-large')  # Figure title
    plt.savefig(saved_path + '/taylor_diagram.png', dpi=300)
    return fig


def best_pipelines(data):
    result = pipelines(data).sort_values(by="meanrank", ascending=True).head(1)
    return result.reset_index()


def reg_plot(saved_path, data, train_size, target):
    best_pipeline = best_pipelines(data=data)
    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1],
                                                        train_size=train_size,
                                                        random_state=0)
    train = pd.concat([x_train, y_train], axis=1).reset_index().drop('index', axis=1)
    test = pd.concat([x_test, y_test], axis=1).reset_index().drop('index', axis=1)

    scaler = StandardScaler()
    scaler.fit(train.iloc[:, :-1])

    scaled_train = scaler.transform(train.iloc[:, :-1])
    scaled_test = scaler.transform(test.iloc[:, :-1])

    scaled_train = pd.concat(
        [pd.DataFrame(scaled_train, columns=train.iloc[:, :-1].columns), train.iloc[:, -1]], axis=1)
    scaled_test = pd.concat(
        [pd.DataFrame(scaled_test, columns=test.iloc[:, :-1].columns), test.iloc[:, -1]], axis=1)

    model = joblib.load(saved_path + '/results/{}'.format(best_pipeline.loc[0, 'Regressor']))

    if target == 'train':
        y_train_pred = pd.DataFrame(
            model[0].predict(scaled_train.loc[:, eval(best_pipeline.loc[0, 'Optimal features'])]))
        y_train = scaled_train.iloc[:, -1]
        pred_test = pd.concat([y_train_pred, y_train], axis=1)
        pred_test.columns = ['Predicted Transaction Price', 'Actual Transaction Price']
        ax = sns.regplot(x='Actual Transaction Price',
                         y='Predicted Transaction Price',
                         scatter_kws={'s': 7, 'alpha': 0.5},
                         data=pred_test)
        plt.title('Train Set: $R^{}$={}'.format(2, '%.2f' % r2_score(y_train, y_train_pred)))
        return ax

    else:
        y_test_pred = pd.DataFrame(model[0].predict(scaled_test.loc[:, eval(best_pipeline.loc[0, 'Optimal features'])]))
        y_test = scaled_test.iloc[:, -1]
        pred_test = pd.concat([y_test_pred, y_test], axis=1)
        pred_test.columns = ['Predicted Transaction Price', 'Actual Transaction Price']
        ax = sns.regplot(x='Actual Transaction Price',
                         y='Predicted Transaction Price',
                         scatter_kws={'s': 7, 'alpha': 0.5},
                         data=pred_test)
        plt.title('Test Set: $R^{}$={}'.format(2, '%.2f' % r2_score(y_test, y_test_pred)))
        return ax


def nemenyi_test_plot(performance_data):
    performance_data['%RMSE'] = performance_data['%RMSE'].str.replace('%', '').astype(float)
    performance_data['MAPE'] = performance_data['MAPE'].str.replace('%', '').astype(float)
    df = performance_data[['Number of features', 'Model', 'R2']]

    avg_rank = df.groupby('Number of features').R2.rank(ascending=False).groupby(df.Model).mean()
    test_results = sp.posthoc_nemenyi_friedman(
        df,
        melted=True,
        block_col='Number of features',
        group_col='Model',
        y_col='R2')

    plt.figure(figsize=(10, 2), dpi=100)
    plt.title('Critical difference diagram of base models in AutoML4RPV')
    sp.critical_difference_diagram(avg_rank, test_results)
    plt.savefig(f'./paper_test/{region}_test/nemenyi_test.png', format='PNG', dpi=500, bbox_inches='tight', pad_inches=+0.1)
    plt.show()


if __name__ == "__main__":
    # mrmr_performance(folder_path='./results')
    # mrmr_performance_plot(metrics='R2')
    # leaderboard(metrics='R2')
    # wilcoxon_paired_matrix_plot()
    # dia = taylor_diagram(pipeline_id=['Pipeline_1', 'Pipeline_3'])
    # print(best_pipelines(n=3))
    # print(best_pipelines(n=3).reset_index().loc[0, 'Pipeline_ID'])
    # reg_plot(data=pd.read_csv('./test/train.csv'),
    #          train_size=0.75,
    #          target='train')
    # data=pd.read_csv('./paper_test/ny_test/benchmark_results/pipeline_results.csv')
    # nemenyi_test_plot(data=data)
    # print(best_pipelines(data=data).reset_index().loc[0, 'Pipeline_ID'])
    region = 'sg'
    wilcoxon_paired_matrix_plot(region)


