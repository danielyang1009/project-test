import numpy as np
import pandas as pd
import joblib as jl
import os
import multiprocessing
import shutil
from numba import prange,njit
import scipy.stats as stats

def _ols_np(data,vars):
    #assert np.linalg.matrix_rank(data[vars[1]])==len(vars[1]), 'Independent variable matrix is not full column rank. Check data carefully.'
    gamma,_,_,_ = np.linalg.lstsq(data[vars[1]],data[vars[0]],rcond=None)
    return pd.Series(gamma.flatten())
    
def fama_macbeth(data,t,yvar,xvar,intercept=True):
    """
    Basic Fama Macbeth regression implementation with regressions performed by :code:`numpy` linear algebra routines and 
    grouping performed by :code:`pandas` groupby functionality.
    
    Parameters
    ----------
    
    data: pandas.core.frame.DataFrame
        A dataframe with regressand, regressors, and time variable for Fama Macbeth regression.
    t: str
        The name of the time variable in :code:`data`. 
    yvar: str
        The name of the regressand variable in :code:`data`. 
    xvar: list(str)
        A list of the names of the regressor variables in :code:`data`. 
    intercept: bool
        Whether or not to regress with an intercept.
        
    Returns 
    -------
    
    A pandas DataFrame which contains regression coefficients for each time period in :code:`data`.
    
    """
    
    _assertions(data,t,yvar,xvar,intercept)
    
    if intercept:
        data['intercept'] = 1
        xvar.insert(0,'intercept')
    d = (data.groupby(t).apply(_ols_np,[yvar,xvar]))
    d.columns = xvar
    return d
    
def _assertions(data,t,yvar,xvar,intercept,n_jobs=-999,backend=-999,memmap=-999,parallel=-999):

    assert isinstance(data,pd.core.frame.DataFrame), 'Invalid input for `data`.'
    assert isinstance(t,str), 'Invalid input for `t`.'
    assert isinstance(xvar,list), 'Invalid input for `xvar`.'
    assert isinstance(intercept,bool), 'Invalid input for `intercept`.'
    assert hasattr(yvar,'__iter__')==False or isinstance(yvar,str), 'Invalid input for `yvar`. `yvar` must either be a string or not be iterable'
    assert n_jobs==-999 or isinstance(n_jobs,int), 'Invalid input for `n_jobs`.'
    assert backend==-999 or (backend in ['loky','multiprocessing','threading']), 'Invalid input for `backend`.'
    assert memmap==-999 or isinstance(memmap,bool), 'Invalid input for `memmmap`.'
    assert parallel==-999 or isinstance(parallel,bool), 'Invalid input for `parallel`.'
    
def fama_macbeth_parallel_jl(data,t,yvar,xvar,intercept=True,n_jobs=2,backend='loky',memmap=False):
    """
    Parallel (:code:`joblib`) Fama Macbeth regression implementation with regressions performed by :code:`numpy` linear algebra routines 
    and grouping performed by :code:`pandas` groupby.
    
    Parameters
    ----------
    
    data: pandas.core.frame.DataFrame
        A dataframe with regressand, regressors, and time variable for Fama Macbeth regression.
    t: str
        The name of the time variable in :code:`data`.
    yvar: str
        The name of the regressand variable in :code:`data`. 
    xvar: list(str)
        A list of the names of the regressor variables in :code:`data`. 
    intercept: bool
        Whether or not to regress with an intercept.
    n_jobs: int
        Number of parallel jobs to be run.
    backend: {'loky','multiprocessing','threading'}
        The joblib backend to use for parallel processing. 'loky' is used by default and is recommended.
    memmap: Bool
        Whether to save results to a memory map (saves to new ./joblib_mmap folder). Defaults to False. 
        
    Returns 
    -------
    
    A pandas DataFrame which contains regression coefficients for each time period in :code:`data`.
    
    """
    
    _assertions(data,t,yvar,xvar,intercept,n_jobs=n_jobs,backend=backend,memmap=memmap)
    
    if intercept:
        data['intercept'] = 1
        xvar.insert(0,'intercept')
    
    folder = ".\joblib_mmap"
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass
    
    if memmap:
        data_f_memmap = os.path.join(folder,"data.memmap")
        jl.dump(data.groupby(t),data_f_memmap)
        grouped = jl.load(data_f_memmap,mmap_mode='r')
    else:
        grouped = data.groupby(t)
    # grouped = data.groupby(t)
    
    retlst =  jl.Parallel(n_jobs=n_jobs,backend=backend) \
                (jl.delayed(_ols_np)(group,[yvar,xvar]) for name,group in grouped)
    d = pd.concat(retlst,axis=1).T
    d.columns = xvar
    
    if memmap:
        try:
            shutil.rmtree(folder)
        except:
            print('Could not clean-up memmap folder: {}'.format(folder))
            pass
    
    return d
        
def fm_summary(p,pvalues=False):
    """
    Summary function for Fama Macbeth regression results.
    
    Parameters
    ----------
    
    p: pandas.core.frame.DataFrame
        A DataFrame object returned by a Fama Macbeth regression function in this library.
    pvalues: Boolean
        Whether or not to include p-values in the summary table.
    
    Returns
    -------
    A summary DataFrame with Fama Macbeth standard errors, mean coefficients, t-statistics, and p-values.
    
    """
    assert isinstance(p,pd.core.frame.DataFrame), 'Invalid input for `p`.'
    
    s = p.describe().T
    s['std_error'] = s['std']/np.sqrt(s['count'])
    s['tstat'] = s['mean']/s['std_error']
    s['pval'] = stats.t.sf(np.abs(s['tstat']),s['count']-1)*2 # 2-sided p-value for the t-statistic
    retlst = ['mean','std_error','tstat']
    if pvalues:
        retlst.append('pval')
    return s[retlst]
   
@njit(parallel=False)
def _ols_numba(d,ti,yvari,xvari):
    periods = np.unique(d[:,ti])
    gammas = np.zeros((periods.shape[0],xvari.shape[0]))
    for i in prange(periods.shape[0]):
        dp = d[d[:,ti]==periods[i]]
        #assert np.linalg.matrix_rank(dp[:,xvari])==xvari.shape[0], 'Independent variable matrix is not full column rank. Check data carefully.'
        gamma,nul1,nul2,nul3 = np.linalg.lstsq(dp[:,xvari],dp[:,yvari],rcond=-1)
        gammas[i,:] = gamma
    return gammas
    
@njit(parallel=True)
def _ols_numba_parallel(d,ti,yvari,xvari):
    periods = np.unique(d[:,ti])
    gammas = np.zeros((periods.shape[0],xvari.shape[0]))
    for i in prange(periods.shape[0]):
        dp = d[d[:,ti]==periods[i]]
        #assert np.linalg.matrix_rank(dp[:,xvari])==xvari.shape[0], 'Independent variable matrix is not full column rank. Check data carefully.'
        gamma,nul1,nul2,nul3 = np.linalg.lstsq(dp[:,xvari],dp[:,yvari],rcond=-1)
        gammas[i,:] = gamma
    return gammas        

def fama_macbeth_numba(data,t,yvar,xvar,intercept=True,parallel=False):
    """
    Fama Macbeth regression implementation for small data sets using compiled machine code. 
    
    Parameters
    ----------
    
    data: pandas.core.frame.DataFrame
        A dataframe with regressand, regressors, and time variable for Fama Macbeth regression. This dataframe must have strictly numeric 
        types with the exception of the time variable which may be a :code:`datetime64[ns]` type.
    t: str
        The name of the time variable in :code:`data`. Note that t must be either a numeric type or :code:`datetime64[ns]` type 
        (i.e. via pd.to_datetime()).
    yvar: str
        The name of the regressand variable in :code:`data`. 
    xvar: list(str)
        A list of the names of the regressor variables in :code:`data`. 
    intercept: bool
        Whether or not to regress with an intercept.
    parallel: bool
        Whether or not to use a parallel numba implementation.
        
    Returns 
    -------
    
    A pandas DataFrame which contains regression coefficients for each time period in :code:`data`.
    
    """
    
    _assertions(data,t,yvar,xvar,intercept,parallel)
    _numba_assertions(data,t)
    data = _numeric_t(data,t)
    
    if intercept==True:
        data['intercept'] = 1
        xvar.insert(0,'intercept')
    ti = data.columns.get_loc(t)
    yvari = data.columns.get_loc(yvar)
    xvari = [data.columns.get_loc(x) for x in xvar]
    d = data.to_numpy()
    if parallel:
        gammas = _ols_numba_parallel(d,ti,int(yvari),np.array(xvari))
    else:
        gammas = _ols_numba(d,ti,int(yvari),np.array(xvari))
    result = pd.DataFrame(gammas,columns=xvar,index=data[t].unique())
    return result
    
def _numba_assertions(data,t):
    if len(data.drop(t,axis=1).dtypes) > 0:
        assert all([pd.api.types.is_numeric_dtype(d) for d in data.drop(t,axis=1).dtypes]), 'All columns except the time column must have a numeric type.'
    else:
        assert pd.api.types.is_numeric_dtype(data.drop(t,axis=1).dtypes), 'All columns except the time column must have a numeric type.'
    
def _numeric_t(data,t):
    if data[t].dtype=='<M8[ns]':
        data['reftime'] = pd.to_datetime('1900-01-01')
        data[t] = (data[t]-data['reftime']).dt.total_seconds().astype(int)
        data = data.drop('reftime',axis=1)
    return data
    
def fama_macbeth_numba_joblib_parallel(data,t,yvar,xvar,intercept=True,n_jobs=2,backend='loky'):
    """
    
    Parameters
    ----------
    
    data: pandas.core.frame.DataFrame
        A dataframe with regressand, regressors, and time variable for Fama Macbeth regression. This dataframe must have strictly numeric 
        types with the exception of the time variable which may be a :code:`datetime64[ns]` type.
    t: str
        The name of the time variable in :code:`data`. Note that t must be either a numeric type or :code:`datetime64[ns]` type 
        (i.e. via pd.to_datetime()).
    yvar: str
        The name of the regressand variable in :code:`data`. 
    xvar: list(str)
        A list of the names of the regressor variables in :code:`data`. 
    intercept: bool
        Whether or not to regress with an intercept.
    n_jobs: int
        Number of parallel jobs to be run.
    backend: {'loky','multiprocessing','threading'}
        The joblib backend to use for parallel processing. 'loky' is used by default and is recommended.
        
    Returns 
    -------
    
    A pandas DataFrame which contains regression coefficients for each time period in :code:`data`.
    
    """
        
    _assertions(data,t,yvar,xvar,intercept)
    _numba_assertions(data,t)
    data = _numeric_t(data,t)
    
    if intercept==True:
        data['intercept'] = 1
        xvar.insert(0,'intercept')
    ti = data.columns.get_loc(t)
    yvari = data.columns.get_loc(yvar)
    xvari = [data.columns.get_loc(x) for x in xvar]
    grouped = [np.array(g) for n,g in data.groupby(t)]
    retlst = jl.Parallel(n_jobs=n_jobs,backend=backend) \
                (jl.delayed(_ols_numba2)(g,int(ti),int(yvari),np.array(xvari)) for g in grouped)
    d = pd.DataFrame(np.vstack(retlst))
    d.columns = xvar
    return d
    
@njit(parallel=False)
def _ols_numba2(d,ti,yvari,xvari):
    #assert np.linalg.matrix_rank(d[:,xvari])==xvari.shape[0], 'Independent variable matrix is not full column rank. Check data carefully.'
    gamma,nul1,nul2,nul3 = np.linalg.lstsq(d[:,xvari],d[:,yvari],rcond=-1)
    return gamma

def fama_macbeth_parallel(data,t,yvar,xvar,intercept=True,backend='loky'):
    """
    Fama Macbeth regression implementation using :code:`pandas` groupby for grouping, linear algebra routines
    compiled with :code:`numba` for regressions, and :code:`joblib` for parallelization. Jobs are pre-dispatched 
    to each core for performance.
    
    Parameters
    ----------
    
    data: pandas.core.frame.DataFrame
        A dataframe with regressand, regressors, and time variable for Fama Macbeth regression. This dataframe must have strictly numeric 
        types with the exception of the time variable which may be a :code:`datetime64[ns]` type.
    t: str
        The name of the time variable in :code:`data`. Note that t must be either a numeric type or :code:`datetime64[ns]` type 
        (i.e. via pd.to_datetime()).
    yvar: str
        The name of the regressand variable in :code:`data`. 
    xvar: list(str)
        A list of the names of the regressor variables in :code:`data`. 
    intercept: bool
        Whether or not to regress with an intercept.
    backend: {'loky','multiprocessing','threading'}
        The joblib backend to use for parallel processing. 'loky' is used by default and is recommended.
        
    Returns 
    -------
    
    A pandas DataFrame which contains regression coefficients for each time period in :code:`data`.
    
    """
        
    _assertions(data,t,yvar,xvar,intercept)
    _numba_assertions(data,t)
    data = _numeric_t(data,t)
    
    if intercept==True:
        data['intercept'] = 1
        xvar.insert(0,'intercept')
    ti = data.columns.get_loc(t)
    yvari = data.columns.get_loc(yvar)
    xvari = [data.columns.get_loc(x) for x in xvar]
    grouped = [np.array(g) for n,g in data.groupby(t)]
    n_jobs = multiprocessing.cpu_count()
    batch_size = int(len(grouped) / n_jobs) 
    retlst = jl.Parallel(n_jobs=n_jobs,backend=backend,batch_size=batch_size,pre_dispatch='all') \
                (jl.delayed(_ols_numba2)(g,int(ti),int(yvari),np.array(xvari)) for g in grouped)
    d = pd.DataFrame(np.vstack(retlst))
    d.columns = xvar
    return d
    
def fama_macbeth_master(data,t,yvar,xvar,intercept=True):
    """
    Master function for Fama Macbeth regressions which uses the best implementation based on 
    the size of the data provided.
    
    Parameters
    ----------
    
    data: pandas.core.frame.DataFrame
        A dataframe with regressand, regressors, and time variable for Fama Macbeth regression. This dataframe must have strictly numeric 
        types with the exception of the time variable which may be a :code:`datetime64[ns]` type.
    t: str
        The name of the time variable in :code:`data`. Note that t must be either a numeric type or :code:`datetime64[ns]` type 
        (i.e. via pd.to_datetime()).
    yvar: str
        The name of the regressand variable in :code:`data`. 
    xvar: list(str)
        A list of the names of the regressor variables in :code:`data`. 
    intercept: bool
        Whether or not to regress with an intercept.
        
    Returns 
    -------
    
    A pandas DataFrame which contains regression coefficients for each time period in :code:`data`.
    
    """
    
    if all([d < 3.0e2 for d in data.shape]):
        d = fama_macbeth_numba(data,t,yvar,xvar,intercept=intercept)
    else:
        d = fama_macbeth_parallel(data,t,yvar,xvar,intercept=intercept)
    return d