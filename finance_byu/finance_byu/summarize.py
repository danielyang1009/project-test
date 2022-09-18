import scipy.stats as stats
import numpy as np
import pandas as pd

def summary(df):
    """
    Compute a summary for a dataframe of portfolios.

    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        DataFrame with portfolio returns in each column.
  
    Returns
    -------
    output: pandas.core.frame.DataFrame
        Summary DataFrame including t-statistics testing that the mean returns for each portfolio are zero, 
        the count for each portfolio, the mean return for each portfolio, the standard deviation for each 
        portfolio, and quartile returns for each portfolio. 

    """
    s = df.describe().T
    s['tstat'] = s['mean']/(s['std']/np.sqrt(s['count'])) # t-statistic testing that the mean return is zero
    s['pval'] = stats.t.sf(np.abs(s['tstat']),s['count']-1)*2 # 2-sided p-value for the t-statistic
    return s[['count','mean','std','tstat','pval','min','25%','50%','75%','max']].T
    
def summarize(df):
    """
    Compute a summary for a dataframe of portfolios.

    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        DataFrame with portfolio returns in each column.
  
    Returns
    -------
    output: pandas.core.frame.DataFrame
        Summary DataFrame including t-statistics testing that the mean returns for each portfolio are zero, 
        the count for each portfolio, the mean return for each portfolio, the standard deviation for each 
        portfolio, and quartile returns for each portfolio. 

    """
    s = df.describe().T
    s['tstat'] = s['mean']/(s['std']/np.sqrt(s['count'])) # t-statistic testing that the mean return is zero
    s['pval'] = stats.t.sf(np.abs(s['tstat']),s['count']-1)*2 # 2-sided p-value for the t-statistic
    return s[['count','mean','std','tstat','pval','min','25%','50%','75%','max']].T