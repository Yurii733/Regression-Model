import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor


data = pd.read_csv("Metrics_Trial_Median.csv")

# Apply log transformation
data['log_reported_rate'] = np.log(data['reported_rate'])
data['log_median_avg_cite3'] = np.log(data['median_avg_cite3'])
data['log_median_avg_median_age'] = np.log(data['median_avg_median_age'])

# Define independent and dependent variables
X = sm.add_constant(data[['log_median_avg_cite3', 'log_median_avg_median_age']])
Y = data['log_reported_rate']

# Fit the model
model = sm.OLS(Y, X).fit()

# Assumption 1: Linearity (Log-Log Relationship)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.regplot(x='log_median_avg_cite3', y='log_reported_rate', data=data, line_kws={'color': 'red'})
plt.xlabel('log(median_avg_cite3)')
plt.ylabel('log(reported_rate)')
plt.title('Log-Log: log(median_avg_cite3) vs log(reported_rate)')

plt.subplot(1, 2, 2)
sns.regplot(x='log_median_avg_median_age', y='log_reported_rate', data=data, line_kws={'color': 'red'})
plt.xlabel('log(median_avg_median_age)')
plt.ylabel('log(reported_rate)')
plt.title('Log-Log: log(median_avg_median_age) vs log(reported_rate)')
plt.show()

# Assumption 2: Independence of Errors (Durbin-Watson Test)
dw_statistic = sm.stats.stattools.durbin_watson(model.resid)
print(f'Durbin-Watson statistic: {dw_statistic}')

# Assumption 3: Homoscedasticity (Residuals vs Fitted Values)
plt.figure(figsize=(8, 6))
sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, line_kws={'color': 'red'})
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values (Homoscedasticity Check)')
plt.show()

# Assumption 4: Normality of Residuals
sns.histplot(model.resid, kde=True)
plt.title('Histogram of Residuals (Normality Check)')
plt.show()

sm.qqplot(model.resid, line='45')
plt.title('Q-Q Plot of Residuals (Normality Check)')
plt.show()

# Assumption 5: Multicollinearity (Variance Inflation Factor - VIF)
vif_data = pd.DataFrame({'Feature': X.columns, 'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})
print(vif_data)

# Mean of residuals
print(f'Mean of Residuals: {np.mean(model.resid)}')


print(model.summary())
