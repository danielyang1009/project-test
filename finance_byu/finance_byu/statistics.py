import numpy as np
import scipy.stats as stats
import pandas as pd
import statsmodels.formula.api as smf
from finance_byu.regtables import Regtable

def GRS(df,exret,factors):
    """
    Implementation of the Gibbons, Ross, and Shanken (GRS) test for a factor model.
    
    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        DataFrame containing a columns with excess returns and columns with factor returns.
    exret: list
        List of the variable / column names in :code:`df` which are excess portfolio returns to be tested.
    factors: list
        List of the variable / column names in :code:`df` which are factor returns.
    
    Returns
    -------
    output: tuple(float,float,Regtable)
        Tuple containing the GRS statistic, associated p-value, and Regtable with regression results for 
        regressions of the form exret = alpha + summation(beta*factor) + e.   
    
    """
    
    # TODO: Optimize with numba for speed
    
    assert isinstance(df,pd.core.frame.DataFrame), 'df must be a DataFrame'
    assert isinstance(exret,list), 'exret must be a list'
    assert isinstance(factors,list), 'factors must be a list'
    assert all([(r in df.columns) for r in exret]), 'at least one of the entries in exret is not a column / variable name in df'
    assert all([(f in df.columns) for f in factors]), 'at least one of the entries in factors is not a column / variable name in df'
    
    regformulas = [str(r)+' ~ '+' + '.join(factors) for r in exret]
    reg = [smf.ols(f,df).fit() for f in regformulas]
    tbl = Regtable(reg)
    alpha = tbl.get_coefficients().loc['Intercept',:].values # numpy vector
    epsilon = pd.concat([r.predict()-df[exret[i]] for i,r in enumerate(reg)],axis=1).values
    F = df[factors].values
    T = df.shape[0]
    L = len(factors)
    N = len(exret)
    Fave = pd.DataFrame([df[factors].mean().values for i in range(T)]).values
    bigE = np.matmul(epsilon.T,epsilon)/(T-L-1)
    mu = df[factors].mean().values
    fdif = F - Fave
    omega = np.matmul(fdif.T,fdif)/(T-1)
    GRS = (T/N)*((T-N-L)/(T-L-1))*np.matmul(np.matmul(alpha,np.linalg.inv(bigE)),alpha)/(1+np.matmul(np.matmul(mu,np.linalg.inv(omega)),mu))
    pval = stats.f.sf(GRS, N, T-N-L)
    return (GRS,pval,tbl)
