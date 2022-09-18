import numpy as np
from numba import njit,jit,prange
import pandas as pd

@njit(parallel=False)
def _check_minp(win,minp,N):
    if minp > win:
        raise ValueError('minp must be <= win')
    elif minp > N:
        minp = N + 1
    elif minp == 0:
        minp = 1
    elif minp < 0:
        raise ValueError('minp must be >= 0')
    return minp

def _assertions_1(input,win,minp,ddof=-999,errors='raise'):
    assert (isinstance(input,pd.core.series.Series) or isinstance(input,pd.core.series.Series)), 'Input must be a pandas Series or DataFrame'
    if len(input.dtypes) > 0:
        assert all([pd.api.types.is_numeric_dtype(d) for d in input.dtypes]), 'All columns must have a numeric type.'
    else:
        assert pd.api.types.is_numeric_dtype(input.dtypes), 'All columns must be a numeric type.'
    eflg = False
    if errors=='raise':
        assert ddof<input.shape[0], 'Invalid input for ddof: ddof must be less than the number of observations in input'
    elif errors=='return':
        if ddof>=input.shape[0]:
            eflg = True
    else:
        assert errors in ['raise','return'], 'Invalid input for errors.'
    return eflg


def _assertions_2(x,y,win,minp,idx,ddof=-999,errors='raise'):
    assert isinstance(x,pd.core.series.Series), 'x must be a pandas Series'
    assert isinstance(y,pd.core.series.Series), 'y must be a pandas Series'
    assert x.shape==y.shape, 'x and y must have the same shape'
    assert (idx=='x' or idx=='y'), 'Invalid input for idx'
    assert all([pd.api.types.is_numeric_dtype(d) for d in [x.dtype,y.dtype]]), 'All series must have a numeric type.'
    eflg = False
    if errors=='raise':
        assert ddof<x.shape[0], 'Invalid input for ddof: ddof must be less than the number of observations in x and y'
    elif errors=='return':
        if ddof>=x.shape[0]:
            eflg = True
    else:
        assert errors in ['raise','return'], 'Invalid input for errors.'
    return eflg

# def _assertions(input,win,minp,ddof=-999):
    # assert (isinstance(input,pd.core.frame.DataFrame) or isinstance(input,pd.core.series.Series)), 'Input must be a pandas Series or DataFrame'
    # assert ddof<input.shape[0], 'Invalid input for ddof: ddof must be less than the number of observations in input'
    # if len(input.dtypes) > 0:
        # assert all([pd.api.types.is_numeric_dtype(d) for d in input.dtypes]), 'All columns must have a numeric type.'
    # else:
        # assert pd.api.types.is_numeric_dtype(input.dtypes), 'All columns must be a numeric type.'

# def _assertions2(x,y,win,minp,idx,ddof=-999):
    # assert isinstance(x,pd.core.series.Series), 'x must be a pandas Series'
    # assert isinstance(y,pd.core.series.Series), 'y must be a pandas Series'
    # assert ddof<x.shape[0], 'Invalid input for ddof: ddof must be less than the number of observations in x and y'
    # assert x.shape==y.shape, 'x and y must have the same shape'
    # assert (idx=='x' or idx=='y'), 'Invalid input for idx'
    # assert all([pd.api.types.is_numeric_dtype(d) for d in [x.dtype,y.dtype]]), 'All series must have a numeric type.'


def roll_sum(input,win,minp,errors='raise'):
    """
    Computes the rolling sum for a :code:`pandas` Series.

    Parameters
    ----------
    input: pandas.core.series.Series
        This series must have strictly numeric type.
    win: int
        Length of the moving window
    minp: int
        Minimum number of observations in window required to have a value 
        (otherwise result is NaN).
    errors: {'raise','return'}
        Whether to raise an error and stop running or to return NaN on manageable errors. 'return' 
        is suggested for running in a groupby in which there may be some groups which 
        do not have a sufficient number of observations.

    Returns
    -------
    output: pandas.core.series.Series
        A :code:`pandas` Series with the rolling sum of :code:`input`.
    """
    eflg = _assertions_1(input,win,minp,errors=errors)
    if eflg:
        r = pd.Series(np.nan,index=input.index)
    else:
        r = pd.Series(_roll_sum(input.values,win,minp),input.index)
    return r

@njit(parallel=False)
def _roll_sum(input,win,minp):
    """
    Computes the rolling sum of a :code:`numpy` array.

    Parameters
    ----------
    input: np.array of type double
    win: int
        Length of the moving window
    minp: int
        Minimum number of observations in window required to have a value 
        (otherwise result is NaN).

    Returns
    -------
    output: np.array of type double
    """

    val = float(0) 
    prev = float(0)
    sum_x = float(0)
    nobs = int(0)
    N = int(len(input))

    minp = _check_minp(win, minp, N)   
    output = np.empty(N,dtype=np.float64)

    for i in range(minp - 1):
        val = input[i]

        if val == val:
            nobs += 1
            sum_x += val

        output[i] = np.nan
        

    for i in range(minp - 1,N):        
        val = input[i]

        if val == val:        
            nobs += 1
            sum_x += val

        if i > win - 1:
            prev = input[i - win]
            
            if prev == prev:
                sum_x -= prev                
                nobs -= 1

        if nobs >= minp:
            output[i] = sum_x
        else:
            output[i] = np.nan            

    return output            

def roll_mean(input,win,minp,errors='raise'):
    """
    Computes the rolling mean of a :code:`pandas` Series.

    Parameters
    ----------
    input: pandas.core.series.Series
        This series must have strictly numeric type.
    win: int
        Length of the moving window
    minp: int
        Minimum number of observations in window required to have a value 
        (otherwise result is NaN).
    errors: {'raise','return'}
        Whether to raise an error and stop running or to return NaN on manageable errors. 'return' 
        is suggested for running in a groupby in which there may be some groups which 
        do not have a sufficient number of observations.

    Returns
    -------
    output: pandas.core.series.Series
        A :code:`pandas` Series with the rolling mean of :code:`input`.
    """

    eflg = _assertions_1(input,win,minp,errors=errors)
    if eflg:
        r = pd.Series(np.nan,index=input.index)
    else:
        r = pd.Series(_roll_mean(input.values,win,minp),input.index)
    return r

@njit(parallel=False)
def _roll_mean(input,win,minp):
    """
    Computes the rolling mean of a numpy array.

    Parameters
    ----------
    input: np.array of type double
    win: int 
        Length of the moving window
    minp: int
        Minimum number of observations in window required to have a value 
        (otherwise result is NaN).

    Returns
    -------
    output: np.array of type double
    """

    val = float(0)
    prev = float(0)
    sum_x = float(0)
    nobs = int(0)
    N = int(len(input))

    minp = _check_minp(win, minp, N)   
    output = np.empty(N,dtype=np.float64)

    for i in range(minp - 1):
        val = input[i]

        if val == val:
            nobs += 1
            sum_x += val

        output[i] = np.nan
        
    for i in range(minp - 1,N):        
        val = input[i]

        if val == val:        
            nobs += 1
            sum_x += val

        if i > win - 1:
            prev = input[i - win]
            
            if prev == prev:
                sum_x -= prev                
                nobs -= 1

        if nobs >= minp:
            output[i] = sum_x/nobs
        else:
            output[i] = np.nan            

    return output            


def roll_var(input,win,minp,ddof=1,errors='raise'):
    """
    Computes the rolling variance for a :code:`pandas` Series.

    Parameters
    ----------
    input: pandas.core.series.Series
        This series must have strictly numeric type.
    win: int
        Length of the moving window
    minp: int
        Minimum number of observations in window required to have a value 
        (otherwise result is NaN).
    ddof: int
        Delta degrees of freedom. The divisor used in calculations is N - ddof, 
        where N represents the number of elements.
    errors: {'raise','return'}
        Whether to raise an error and stop running or to return NaN on manageable errors. 'return' 
        is suggested for running in a groupby in which there may be some groups which 
        do not have a sufficient number of observations.

    Returns
    -------
    output: pandas.core.series.Series
        A :code:`pandas` Series with the rolling variance of :code:`input`.
    """
    
    eflg = _assertions_1(input,win,minp,ddof=ddof,errors=errors)
    if eflg:
        r = pd.Series(np.nan,index=input.index)
    else:
        r = pd.Series(_roll_var(input.values,win,minp,ddof=ddof),input.index)
    return r


@njit(parallel=False)
def _roll_var(input,win,minp,ddof=1):
    """
    Computes the rolling variance of a numpy array.

    Parameters
    ----------
    input: np.array of type double
    win: int 
        Length of the moving window
    minp: int
        Minimum number of observations in window required to have a value 
        (otherwise result is NA).
    ddof: int
        Delta degrees of freedom. The divisor used in calculations is N - ddof, 
        where N represents the number of elements.

    Returns
    -------
    output: np.array of type double
    """

    val = float(0)
    prev = float(0)
    sum_x = float(0)
    sum_xx = float(0)
    nobs = int(0)
    N = int(len(input))

    minp = _check_minp(win, minp, N)        
    output = np.empty(N,dtype=np.float64)

    for i in range(minp - 1):
        val = input[i]

        if val == val:
            nobs += 1
            sum_x += val
            sum_xx += val**2

        output[i] = np.nan


    for i in range(minp - 1,N):        
        val = input[i]

        if val == val:
            nobs += 1
            sum_x += val
            sum_xx += val**2

        if i > win - 1:
            prev = input[i - win]
            
            if prev == prev:
                sum_x -= prev
                sum_xx -= prev**2
                nobs -= 1

        if nobs >= minp:
            # pathological case
            if nobs == 1:
                output[i] = 0
                continue

            val = (nobs * sum_xx - sum_x**2) / (nobs * (nobs - ddof))
            if val < 0:
                val = 0

            output[i] = val
        else:
            output[i] = np.nan

    return output

def roll_std(input,win,minp,ddof=1,errors='raise'):
    """
    Computes the rolling standard deviation for a :code:`pandas` Series.

    Parameters
    ----------
    input: pandas.core.series.Series
        This series must have strictly numeric type.
    win: int
        Length of the moving window
    minp: int
        Minimum number of observations in window required to have a value 
        (otherwise result is NaN).
    ddof: int
        Delta degrees of freedom. The divisor used in calculations is N - ddof, 
        where N represents the number of elements.
    errors: {'raise','return'}
        Whether to raise an error and stop running or to return NaN on manageable errors. 'return' 
        is suggested for running in a groupby in which there may be some groups which 
        do not have a sufficient number of observations.

    Returns
    -------
    output: pandas.core.series.Series
        A :code:`pandas` Series with the rolling standard deviation of :code:`input`.
    """
    
    eflg = _assertions_1(input,win,minp,ddof=ddof,errors=errors)
    if eflg:
        r = pd.Series(np.nan,index=input.index)
    else:
        r = pd.Series(_roll_std(input.values,win,minp,ddof=ddof),input.index)
    return r


@njit(parallel=False)
def _roll_std(input,win,minp,ddof=1):
    """
    Computes the rolling standard deviation of a numpy array.

    Parameters
    ----------
    input: np.array of type double
    win: int 
        Length of the moving window
    minp: int 
        Minimum number of observations in window required to have a value 
        (otherwise result is NA).
    ddof: int
        Delta cegrees of freedom. The divisor used in calculations is N - ddof, 
        where N represents the number of elements.

    Returns
    -------
    output: np.array of type double
    """

    val = float(0)
    prev = float(0)
    sum_x = float(0)
    sum_xx = float(0)
    nobs = int(0)
    N = int(len(input))

    minp = _check_minp(win, minp, N)        
    output = np.empty(N,dtype=np.float64)

    for i in range(minp - 1):
        val = input[i]
        
        if val == val:
            nobs += 1
            sum_x += val
            sum_xx += val**2

        output[i] = np.nan

    for i in range(minp - 1,N):        
        val = input[i]

        if val == val:
            nobs += 1
            sum_x += val
            sum_xx += val**2

        if i > win - 1:
            prev = input[i - win]
            
            if prev == prev:
                sum_x -= prev
                sum_xx -= prev**2
                nobs -= 1

        if nobs >= minp:
            # pathological case
            if nobs == 1:
                output[i] = 0
                continue

            val = (nobs * sum_xx - sum_x**2) / (nobs * (nobs - ddof))
            if val < 0:
                val = 0

            output[i] = np.sqrt(val)
        else:
            output[i] = np.nan

    return output

def roll_cov(x,y,win,minp,ddof=1,idx='x',errors='raise'):
    """
    Computes the rolling covariance of two :code:`pandas` series.

    Parameters
    ----------
    x: pandas.core.series.Series
        This series must have strictly numeric type.
    y: pandas.core.series.Series
        This series must have strictly numeric type.
    win: int
        Length of the moving window
    minp: int
        Minimum number of observations in window required to have a value 
        (otherwise result is NaN).
    ddof: int
        Delta degrees of freedom. The divisor used in calculations is N - ddof, 
        where N represents the number of elements.
    idx: {'x','y'}
        Whether to use the index for x or for y for the return series. Defaults to 'x'.
    errors: {'raise','return'}
        Whether to raise an error and stop running or to return NaN on manageable errors. 'return' 
        is suggested for running in a groupby in which there may be some groups which 
        do not have a sufficient number of observations.

    Returns
    -------
    output: pandas.core.series.Series
        A :code:`pandas` Series with the rolling covariance of x and y.
    """
    
    eflg = _assertions_2(x,y,win,minp,idx,ddof=ddof,errors=errors)
    if idx=='x':
        myidx = x.index
    else: 
        myidx = y.index
    if eflg:
        r = pd.Series(np.nan,index=myidx)
    else:
        r = pd.DataFrame(_roll_cov(x.values,y.values,win,minp,ddof=ddof),myidx)
    return r

@njit(parallel=False)
def _roll_cov(arr_x,arr_y,win,minp,ddof=1):
    """
    Computes the rolling covariance between two numpy arrays

    Parameters
    ----------
    arr_x: np.array of type double
    arr_y : np.array of type double
    win: int 
        Length of the moving window
    minp: int
        Minimum number of observations in window required to have a value 
        (otherwise result is NA).
    ddof: int
        Delta degrees of freedom. The divisor used in calculations is N - ddof, 
        where N represents the number of elements.

    Returns
    -------
    output: np.array of type double
    """

    sum_x = float(0)
    sum_y = float(0)
    sum_xy = float(0)
    nobs = int(0)
    N = int(len(arr_x))

    minp = _check_minp(win, minp, N)        
    output = np.empty(N,dtype=np.float64)

    for i in range(minp - 1):    
        x = arr_x[i]
        y = arr_y[i]        

        if x == x and y == y:
            nobs += 1
            sum_x += x
            sum_y += y
            sum_xy += x*y

        output[i] = np.nan

    for i in range(minp - 1,N):        
        x = arr_x[i]
        y = arr_y[i]        

        if x == x and y == y:
            nobs += 1
            sum_x += x
            sum_y += y
            sum_xy += x*y

        if i > win - 1:
            prev_x = arr_x[i - win]
            prev_y = arr_y[i - win]
            
            if prev_x == prev_x and prev_y == prev_y:
                nobs -= 1
                sum_x -= prev_x
                sum_y -= prev_y                
                sum_xy -= prev_x*prev_y

        if nobs >= minp:
            # pathological case
            if nobs == 1:
                output[i] = 0
                continue

            output[i] = (nobs * sum_xy - sum_x*sum_y) / (nobs * (nobs - ddof))
        else:
            output[i] = np.nan

    return output

def roll_idio(y,x,win,minp,ddof=1,idx='x',errors='raise'):
    """
    Computes the rolling idio residual standard deviation from a univariate 
    regression, y = a + bx + e.

    Parameters
    ----------
    y: pandas.core.series.Series
        This series must have strictly numeric type.
    x: pandas.core.series.Series
        This series must have strictly numeric type.
    win: int
        Length of the moving window
    minp: int
        Minimum number of observations in window required to have a value 
        (otherwise result is NaN).
    ddof: int
        Delta degrees of freedom. The divisor used in calculations is N - ddof, 
        where N represents the number of elements.
    idx: {'x','y'}
        Whether to use the index for x or for y for the return series. Defaults to 'x'.
    errors: {'raise','return'}
        Whether to raise an error and stop running or to return NaN on manageable errors. 'return' 
        is suggested for running in a groupby in which there may be some groups which 
        do not have a sufficient number of observations.

    Returns
    -------
    output: pandas.core.series.Series
        A :code:`pandas` Series with the rolling idio residual standard deviation from a univariate regression, y = a + bx + e.
    """
    eflg = _assertions_2(x,y,win,minp,idx,ddof=ddof,errors=errors)
    if idx=='x':
        myidx = x.index
    else: 
        myidx = y.index
    if eflg:
        r = pd.Series(np.nan,index=myidx)
    else:
        r = pd.DataFrame(_roll_idio(y.values,x.values,win,minp,ddof=ddof),myidx)
    return r


@njit(parallel=False)
def _roll_idio(arr_y,arr_x,win,minp,ddof=1):
    """
    Computes the rolling idio residual standard deviation from a univariate 
    regression, y = a + bx + e.

    Parameters
    ----------
    arr_y: np.array, dtype=double
        the dependent variable 
    arr_x: np.array, dtype=double
        the independent variable
    win: float
        Length of the moving window
    minp: float
        Minimum number of observations in window required to have a value 
        (otherwise result is NA).
    ddof: float 
        Delta cegrees of freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.

    Returns
    -------
    out: np.array, dtype = double
    """
    
    var_y = _roll_var(arr_y,win,minp,ddof)
    var_x = _roll_var(arr_x,win,minp,ddof)
    cov   = _roll_cov(arr_y,arr_x,win,minp,ddof)    

    N = int(len(arr_x))
    out = np.empty(N,dtype=np.float64)

    for i in range(0,N):
        out[i] = np.sqrt(var_y[i] - cov[i]**2/var_x[i])

    return out

def roll_beta(y,x,win,minp,ddof=1,idx='x',errors='raise'):
    """
    Computes the rolling estimated slope coefficient (beta)  from an univariate regression, y = a + bx + e. 

    Parameters
    ----------
    y: pandas.core.series.Series
        This series must have strictly numeric type.
    x: pandas.core.series.Series
        This series must have strictly numeric type.
    win: int
        Length of the moving window
    minp: int
        Minimum number of observations in window required to have a value 
        (otherwise result is NaN).
    ddof: int
        Delta degrees of freedom. The divisor used in calculations is N - ddof, 
        where N represents the number of elements.
    idx: {'x','y'}
        Whether to use the index for x or for y for the return series. Defaults to 'x'.
    errors: {'raise','return'}
        Whether to raise an error and stop running or to return NaN on manageable errors. 'return' 
        is suggested for running in a groupby in which there may be some groups which 
        do not have a sufficient number of observations.

    Returns
    -------
    output: pandas.core.series.Series
        A :code:`pandas` Series with the rolling estimated slope coefficient (beta)  from an univariate regression, y = a + bx + e. 
    """
    
    eflg = _assertions_2(x,y,win,minp,idx,ddof=ddof,errors=errors)
    if idx=='x':
        myidx = x.index
    else: 
        myidx = y.index
    if eflg:
        r = pd.Series(np.nan,index=myidx)
    else:
        r = pd.DataFrame(_roll_beta(y.values,x.values,win,minp,ddof=ddof),myidx)
    return r

@njit(parallel=False)
def _roll_beta(arr_y,arr_x,win,minp,ddof=1):
    """
    Computes the rolling estimated slope coefficient (beta)  from an 
    univariate regression, y = a + bx + e. 

    Parameters
    ----------
    arr_y: np.array, dtype=double
        the dependent variable 
    arr_x: np.array, dtype=double
        the independent variable
    win: float
        Length of the moving window
    minp: float
        Minimum number of observations in window required to have a value 
        (otherwise result is NA).
    ddof: float 
        Delta cegrees of freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.

    Returns
    -------
    out: np.array, dtype = double
    """
    
    var_x = _roll_var(arr_x,win,minp,ddof)
    cov   = _roll_cov(arr_y,arr_x,win,minp,ddof)    

    N = int(len(arr_x))
    out = np.empty(N,dtype=np.float64)

    for i in range(N):
        out[i] = cov[i]/var_x[i]

    return out
    
import multiprocessing
import joblib as jl
from numba import njit
    
    
# Acknowledgements: https://stackoverflow.com/questions/36526708/comparing-python-numpy-numba-and-c-for-matrix-multiplication
    
@njit(parallel=False)
def _ols_numba(d,yvari,xvari):
    #assert np.linalg.matrix_rank(d[:,xvari])==xvari.shape[0], 'Independent variable matrix is not full column rank. Check data carefully.'
    gamma,nul1,nul2,nul3 = np.linalg.lstsq(d[:,xvari],d[:,yvari],rcond=-1)
    return gamma
    
@njit(parallel=False)
def _ols_residvol_numba(d,yvari,xvari,count,ddof):
    gamma,nul1,nul2,nul3 = np.linalg.lstsq(d[:,xvari],d[:,yvari],rcond=-1)
    resid = _Ab_numba(d[:,xvari],gamma)
    std = np.sqrt(resid.var()*count/(count-ddof))
    r = list(gamma)
    r.append(resid[-1])
    r.append(std)
    return np.array(r)
    
@njit(parallel=False)
def _ols_resid_numba(d,yvari,xvari):
    gamma,nul1,nul2,nul3 = np.linalg.lstsq(d[:,xvari],d[:,yvari],rcond=-1)
    resid = 0
    for i in xvari:
        resid += gamma[i]*d[-1,i]
    r = list(gamma)
    r.append(resid)
    return np.array(r)
    
@njit(parallel=False)
def _matmul_numba(A,B):
    # Note: This is only for A,B nonvectors 
    
    m, n = A.shape
    p = B.shape[1]

    C = np.zeros((m,p))

    for i in range(0,m):
        for j in range(0,p):
            for k in range(0,n):
                C[i,j] += A[i,k]*B[k,j] 
    return C

@njit(parallel=False)
def _Ab_numba(A,b):
    # Note: This is only for A matrix and b vector.
    
    m, n = A.shape

    c = np.zeros(m)

    for i in range(0,m):
        for k in range(0,n):
            c[i] += A[i,k]*b[k] 
    return c


def _numba_assertions(data):
    if len(data.dtypes) > 0:
        assert all([pd.api.types.is_numeric_dtype(d) for d in data.dtypes]), 'All columns must have a numeric type.'
    else:
        assert pd.api.types.is_numeric_dtype(data.dtypes), 'All columns must have a numeric type.'
        
def _assertions(data,yvar,xvar,intercept,backend,roll,residuals,ddof,predispatch,flagop,append):
    assert isinstance(data,pd.core.frame.DataFrame), 'Invalid input for `data`.'
    assert isinstance(xvar,list), 'Invalid input for `xvar`.'
    assert isinstance(intercept,bool), 'Invalid input for `intercept`.'
    assert hasattr(yvar,'__iter__')==False or isinstance(yvar,str), 'Invalid input for `yvar`. `yvar` must either be a string or not be iterable'
    assert backend in ['loky','multiprocessing','threading'], 'Invalid input for `backend`.'
    assert isinstance(roll,int), 'Invalid input for `roll`. `roll` must be an integer.'
    assert (residuals =='residvol') or isinstance(residuals,bool), "Invalid input for `residuals`. `residuals` must be boolean or 'residvol'." 
    assert (ddof=='default') or (isinstance(ddof,int) and (ddof < roll)), "Invalid input for `ddof`. `ddof` must be 'default' or an integer less than `roll`."    
    assert predispatch in ['all','auto'], "Invalid input for `predispatch`. `predispatch` must be either be 'all' or 'auto'."
    assert flagop in ['nan','elim'], "Invalid input for `flagop`. `flagop` must be either 'elim' or 'nan'."
    assert (append==False) or isinstance(append,list), "`append` must be False or a list."
    if isinstance(append,list):
        assert all([(x in data.columns) for x in append]), "The entries of `append` must be columns of `data`."

def rolling_multiple(data,yvar,xvar,roll,intercept=True,residuals=False,backend='loky',ddof='default',predispatch='all',flagop='nan',append=False):
    """
    Rolling multiple regression implementation using :code:`pandas` groupby for grouping, linear algebra routines
    compiled with :code:`numba` for regressions, and :code:`joblib` for parallelization. Jobs are pre-dispatched 
    to each core for performance.
    
    Parameters
    ----------
    
    data: pandas.core.frame.DataFrame
        A dataframe with regressand and regressors for multiple regression. This dataframe must have strictly numeric 
        types.
    yvar: str
        The name of the regressand variable in :code:`data`. 
    xvar: list(str)
        A list of the names of the regressor variables in :code:`data`. 
    roll: int
        The number of observations (rows in data) over which to roll, inclusive of the current row. 
        For example, if :code:`roll=120`, the rolling regression will include the current observation and the previous
        119 observations.
    intercept: bool
        Whether or not to regress with an intercept.
    residuals: {bool,'residvol'}
        Whether or not to include residuals in the output dataframe (boolean). If :code:`'residvol'` is input, the standard
        deviation of the residuals from the rolling regression will be output. If :code:`True` is input, the residual corresponding
        to the observation will be output.
    backend: {'loky','multiprocessing','threading'}
        The joblib backend to use for parallel processing. 'loky' is used by default and is recommended.
    ddof: {'default',int}
        Delta degrees of freedom. Defaults to the number of x variables (including intercept if intercept=True) if set to 'default'. 
    predispatch: {'all','auto'}
        Whether to pre-dispatch all parallel jobs (recommended for smaller datasets) or to allow joblib to pre-dispatch according 
        to memory (recommended for large datasets).
    flagop: {'nan','elim'}
        Error handling mechanism. :code:`'nan'` returns :code:`'NaN'` values when there are insufficient observations (better functionality still in development).
        :code:`'elim'` eliminates groups for which there are insufficient observations. If :code:`'elim'` is used, the output dataframe will not have the same length
        (number of observations) as :code:`data`.
    append: False or list(str)
        Whether or not to append other columns of :code:`data` (must be numeric type for now) to the output DataFrame. Append should be :code:`False` if no columns
        should be appended. :code:`append` should be a list of of the column names of the columns to be appended to the end DataFrame.
    
    Returns 
    -------
    
    A pandas DataFrame which contains rolling regression coefficients and residuals, if specified.
    
    """
        
    _assertions(data,yvar,xvar,intercept,backend,roll,residuals,ddof,predispatch,flagop,append)
    _numba_assertions(data)
    
    data = data.copy().reset_index(drop=True)
    
    dataflag = False
    
    if intercept==True:
        data['intercept'] = 1
        xvar.insert(0,'intercept')
    
    yvari = data.columns.get_loc(yvar)
    xvari = [data.columns.get_loc(x) for x in xvar]
    if append != False:
        appendi = [data.columns.get_loc(x) for x in append]
    roller = roll-1
    grouped = [np.array(data.loc[j-roller:j,:]) for j in range(roller,data.shape[0])] 
    n_jobs = multiprocessing.cpu_count()
    batch_size = int(len(grouped) / n_jobs) 
    if ddof=='default':
        ddof = len(xvar)
    count = data.shape[0]
    
    if predispatch=='auto' or batch_size <1:
        predispatch = '2*n_jobs'
        batch_size='auto'

    if (data.shape[0] < roll) or any([g.shape[0]<roller for g in grouped]):
        dataflag = True

    if dataflag:
        if flagop=='elim':
            grouped = [x for x in grouped if x.shape[0]>=roller]
            if len(grouped)<1:
                flagop='nan'
            
        if flagop=='nan':
            
            if residuals == False:
                xvar = xvar
            elif residuals == True:
                xvar = xvar+['resid']
            elif residuals == 'residvol':
                xvar = xvar+['resid','residvol']
            if append != False:
                xvar = xvar+append
            d = pd.DataFrame(np.NaN,index=[i for i in range(count)],columns=xvar)
            
    if (dataflag == False) or ((dataflag==True) and (flagop=='elim')):
        if residuals == False:
            retlst = jl.Parallel(n_jobs=n_jobs,backend=backend,batch_size=batch_size,pre_dispatch=predispatch) \
                        (jl.delayed(_ols_numba)(g,int(yvari),np.array(xvari)) for g in grouped)
            d = pd.DataFrame(np.vstack(retlst))
            d.columns = xvar
        elif residuals == True:
            retlst = jl.Parallel(n_jobs=n_jobs,backend=backend,batch_size=batch_size,pre_dispatch=predispatch) \
                        (jl.delayed(_ols_resid_numba)(g,int(yvari),np.array(xvari)) for g in grouped)
            d = pd.DataFrame(np.vstack(retlst))
            d.columns = xvar+['resid']
        elif residuals == 'residvol':
            retlst = jl.Parallel(n_jobs=n_jobs,backend=backend,batch_size=batch_size,pre_dispatch=predispatch) \
                        (jl.delayed(_ols_residvol_numba)(g,int(yvari),np.array(xvari),count,ddof) for g in grouped)
            d = pd.DataFrame(np.vstack(retlst))         
            d.columns = xvar + ['resid','residstd']
        
        if append != False:
            da = pd.DataFrame(np.vstack([g[-1,appendi] for g in grouped]))
            da.columns = append
            d = pd.concat([d,da],axis=1)
        fill = pd.DataFrame(np.NaN,index=[i for i in range(roller)],columns=d.columns)
        if append != False:
            fill.loc[0:roller,append] = data.loc[0:roller,append]
        d = pd.concat([fill,d],axis=0).reset_index(drop=True)
                            
    return d


