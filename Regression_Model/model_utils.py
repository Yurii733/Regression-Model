import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

def log_transform(df):
    df = df.copy()
    df['log_reported_rate'] = np.log(df['reported_rate'])
    df['log_median_avg_cite3'] = np.log(df['median_avg_cite3'])
    df['log_median_avg_median_age'] = np.log(df['median_avg_median_age'])
    return df

def fit_model(df):
    X = df[['log_median_avg_cite3', 'log_median_avg_median_age']]
    X = sm.add_constant(X)
    Y = df['log_reported_rate']
    model = sm.OLS(Y, X).fit()
    return model, X

def get_vif(X):
    vif_data = pd.DataFrame({
        'Feature': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    vif_filtered = vif_data[vif_data['Feature'] != 'const']
    return vif_filtered

def get_durbin_watson(model):
    return durbin_watson(model.resid)

def mean_residuals(model):
    return np.mean(model.resid)
