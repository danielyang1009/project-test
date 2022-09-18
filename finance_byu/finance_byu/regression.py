import patsy
import statsmodels.api as sm
import pandas as pd
import numpy as np


def areg(formula,data=None,absorb=None,cluster=None):
    
    """
    Performs efficient fixed effects regression.

    Parameters
    ----------
    formula: string
        Patsy / R style formula string. Variables in the formula string must be 
        contained in :code:`data`.
    data: pandas.core.frame.DataFrame
        Pandas DataFrame containing data for regression.
    absorb: string
        Categorical variable to be absorbed (must be in :code:`data`)
    cluster: string
        Cluster variable (must be in :code:`data`) 
        
    Returns
    -------
    output: statsmodels.regression.linear_model.RegressionResultsWrapper
        A :code:`statsmodels` regression results object.
    """
    
    y,X = patsy.dmatrices(formula,data,return_type='dataframe')

    ybar = y.mean()
    y = y -  y.groupby(data[absorb]).transform('mean') + ybar
    
    Xbar = X.mean()
    X = X - X.groupby(data[absorb]).transform('mean') + Xbar
    
    reg = sm.OLS(y,X)
    # Account for df loss from FE transform
    reg.df_resid -= (data[absorb].nunique() - 1)
    
    return reg.fit(cov_type='cluster',cov_kwds={'groups':data[cluster]})