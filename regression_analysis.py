import pandas as pd

df = pd.DataFrame()
new_data = pd.DataFrame()

# Good to know
# - C creates a one-hot encoding of the categorical values, Treatment(reference=0) makes the baseline = 0
# - variables in formulas are the  column  names in df

##############################################################################################################
# LINEAR REGRESSION
# https://www.statsmodels.org/stable/generated/statsmodels.formula.api.ols.html

import statsmodels.formula.api as smf
# Creates, fits and then summarise a linear regression model based on columns in df 
mod = smf.ols(formula='y ~ C(x1) * C(x2,  Treatment(reference=0)) + C(x3)', data=df)
res = mod.fit()
print(res.summary())

# We can later predict on other data, but using scipy for prediction this might be better
predictions = res.predict(new_data)

##############################################################################################################
#LOGISTIC REGRESSION
# https://www.statsmodels.org/stable/generated/statsmodels.formula.api.logit.html

import statsmodels.formula.api as smf
# Creates, fits and then summarise a logistic regression model based on columns in df 
mod = smf.logit(formula='y ~ C(x1) * C(x2,  Treatment(reference=0)) + C(x3)', data=df)
res = mod.fit()
print(res.summary())

# We can later predict on other data, but using scipy for prediction this might be better
predictions = res.predict(new_data)

##############################################################################################################
# ACCESS SUMMARY VALUES

# feature names
variables = res.params.index

# coefficients
coefficients = res.params.values

# p-values
p_values = res.pvalues

# standard errors
standard_errors = res.bse.values

#confidence intervals
res.conf_int()

###############################################################################################################
# CORRELATION COEFFICIANTS

import scipy.stats as stats

# Pearson coefficiant
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html#scipy.stats.pearsonr
stats.pearsonr(df["x1"], df["x2"])

# Spearman coefficiant
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html#scipy.stats.spearmanr
stats.spearmanr(df["x1"], df["x2"])