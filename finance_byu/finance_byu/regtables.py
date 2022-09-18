import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.regression

class Regtable:
    """
    
    Tabular output and storage for sets of statsmodels regression objects.

    
    Parameters
    ----------

    regressions: list(statsmodels.regression.linear_model.RegressionResultsWrapper)
        A list of fitted regression result objects from statsmodels.
    regnames: list, optional 
        A list of names or labels corresponding to the regression result objects provided. The length of 
        regnames must match the length of regressions. Defaults to the dependent variable names
        for each regression.
    orientation: {'vertical','horizontal'}, default = 'vertical'
        Specify whether to have the independent variables on the vertical or horizontal axis of 
        the table. 
    stat: {'tstat','se','pvalues'}, default = 'tstat'
        Specify which statistic to report in table. 
    sig: {False, 'stat','coeff'}, default = False
        Whether to denote statistical significance with '*' or other symbols in table output.
        Defaults to False, which does not denote significance in tables. If statistical significance 
        is to be denoted with regression coefficients, set sig = 'coeff'. If statistical 
        significance is to be denoted with the reported statistic, set sig = 'stat'.
    sig_lbls: dict([(symbol,level)]), default = {'\\*\\**':0.01,'\\*\\*':0.05}
        Dictionary containing symbols to signify corresponding significance levels (if sig = True).
        Defaults to {'\\***':0.01,'\\**':0.05} which displays '\\***' with entries which are statistically
        significant at the :math:`\\alpha=0.01` level and displays '**' with entries which are statistically 
        significant at the :math:`\\alpha=0.05` level.
    bfmt: str, default = '.3f'
        Float formatting string for the estimated coefficients.
    sfmt: str, default = '.2f'
        Float formatting string for the stat paramater (e.g., t-statistics or 
        standard errors).
    rsq: bool, default = True
        Whether or not to display R-squared values in the table.
    nobs: bool, default = True
        Whether or not to display the number of observations in the table.
    intercept_name: str, optional
        An optional argument to set a name for regression intercepts in the table.

    Attributes
    ----------
    regressions: list(statsmodels.regression.linear_model.RegressionResultsWrapper)
        A list of fitted regression result objects from statsmodels.
    regnames: list, optional 
        A list of names or labels corresponding to the regression result objects provided. The length of 
        regnames must match the length of regressions. Defaults to the dependent variable names
        for each regression.
    orientation: {'vertical','horizontal'}, default = 'vertical'
        Specify whether to have the independent variables on the vertical or horizontal axis of 
        the table. 
    stat: {'tstat','se','pvalues'}, default = 'tstat'
        Specify which statistic to report in table. 
    sig: {False, 'stat','coeff'}, default = False
        Whether to denote statistical significance with '*' or other symbols in table output.
        Defaults to False, which does not denote significance in tables. If statistical significance 
        is to be denoted with regression coefficients, set sig = 'coeff'. If statistical 
        significance is to be denoted with the reported statistic, set sig = 'stat'.
    sig_lbls: dict([(symbol,level)]), default = {'\\*\\**':0.01,'\\*\\*':0.05}
        Dictionary containing symbols to signify corresponding significance levels (if sig = True).
        Defaults to {'\\***':0.01,'\\**':0.05} which displays '\\***' with entries which are statistically
        significant at the :math:`\\alpha=0.01` level and displays '**' with entries which are statistically 
        significant at the :math:`\\alpha=0.05` level.
    bfmt: str, default = '.3f'
        Float formatting string for the estimated coefficients.
    sfmt: str, default = '.2f'
        Float formatting string for the stat paramater (e.g., t-statistics or 
        standard errors).
    rsq: bool, default = True
        Whether or not to display R-squared values in the table.
    nobs: bool, default = True
        Whether or not to display the number of observations in the table.
    intercept_name: str, optional
        An optional argument to set a name for regression intercepts in the table.

    Methods
    -------
    render()
        Returns the table as a pandas.core.frame.DataFrame object.
    to_latex()
        Returns the table in LaTeX format. Will likely have options for LaTeX formatting in future releases.
    set_reg_names(names)
        Sets the names of the regressions (regnames) to the argument (names). This method can also take string arguments which set
        regnames according to particular conventions. The only currently available such option is 'num-brackets'
        which sets [1],[2],... as names for the regressions.
    get_coefficients(orientation='horizontal')
        Returns a pandas dataframe with numerical values for coefficients (NaN where appropriate). Orientation can either be 
        'horizontal' or 'vertical', which specifies on which axis to place the independent variables.
    get_tstats(orientation='horizontal')
        Returns a pandas dataframe with numerical values for t-statistics testing the null hypothesis that coefficients are zero 
        (NaN where appropriate). Orientation can either be 'horizontal' or 'vertical', which specifies on which axis to place the independent variables.
    get_se(orientation='horizontal')
        Returns a pandas dataframe with numerical values for coefficient standard errors (NaN where appropriate). Orientation can either be 'horizontal' 
        or 'vertical', which specifies on which axis to place the independent variables.
    get_pvalues(orientation='horizontal')
        Returns a pandas dataframe with numerical values for p-values testing the null hypothesis that coefficients are zero (NaN where appropriate). 
        Orientation can either be 'horizontal' or 'vertical', which specifies on which axis to place the independent variables.
    """
    
    def __init__(self, regressions, regnames=None, orientation='vertical',
                 stat='tstat',sig=False,sig_lbls={'***':0.01,'**':0.05},bfmt='.3f',sfmt='.2f',rsq=True,
                 nobs=True,intercept_name=None):
        
        self.orientation = orientation
        self.regnames = regnames
        self.num_reg = len(regressions)
        self.stat = stat
        self.intercept_name = intercept_name
        self.bfmt = bfmt
        self.sfmt = sfmt 
        self.rsq = rsq
        self.nobs = nobs
        self.regressions = regressions
        self.sig = sig
        self.sig_lbls = sig_lbls
        self.out = None
        self._assertions()
        
    def _assertions(self):
        assert self.orientation=='vertical' or self.orientation=='horizontal', "Invalid input for orientation"
        assert self.regnames==None or isinstance(self.regnames,list), "Invalid input for regnames"
        if self.regnames != None:
            assert len(self.regressions)==len(self.regnames), "Length of regnames must be equal to length of regressions"
        assert self.stat=='tstat' or self.stat=='se' or self.stat=='pvalues', "Invalid input for stat"
        assert isinstance(self.intercept_name,str) or self.intercept_name==None, "intercept_name must be a string"
        assert isinstance(self.bfmt,str), "bfmt must be a string"
        assert isinstance(self.sfmt,str), "sfmt must be a string"
        assert isinstance(self.rsq,bool), "rsq must be boolean"
        assert isinstance(self.nobs,bool), "nobs must be boolean"
        assert isinstance(self.regressions,list), "Invalid input for regressions"
        assert all([isinstance(r,statsmodels.regression.linear_model.RegressionResultsWrapper) for r in self.regressions]), "Invalid input for regressions"
        assert self.sig==False or self.sig=='stat' or self.sig=='coeff', "Invalid input for sig"
        assert isinstance(self.sig_lbls,dict), "Invalid input for sig_lbls"
        assert all(isinstance(x,str) for x in self.sig_lbls.keys()), "Keys in sig_lbls must be strings"
        try:
            assert all(((x<=1) and (x>=0)) for x in self.sig_lbls.values()), "Significance levels in sig_lbls must be in [0,1]"
        except:
            raise ValueError('Ensure significance levels in sig_lbls are numeric type')
        
    def _form(self):
        
        sig_ = sorted(self.sig_lbls.items(), key=lambda kv:kv[1])
        self.sigbins = [x[1] for x in sig_]
        self.sigbins.append(1.01)
        self.sigbins.insert(0,-0.01)
        self.siglabels = [x[0] for x in sig_]
        self.siglabels.append('')

        if self.regnames == None:
            cols = [x.model.endog_names for x in self.regressions]    
        else:
            cols = self.regnames
        res = [self._stack(r) for r in self.regressions]
        
        obs = pd.Series([int(r.nobs) for r in self.regressions],cols,name='Obs')
        rsqs = pd.Series(['{:{fmt}}'.format(r.rsquared,fmt=self.sfmt) for r in self.regressions],cols,
                        name='Rsq')

        if self.orientation == 'vertical':
            out = pd.concat(res,axis=1,keys=cols).reset_index()
            out.loc[out['level_1'] == self.stat,'level_0'] = ''
            out.loc[out['level_1'] == (self.stat+'star'),'level_0'] = ''
            out = out.drop('level_1',axis=1).set_index('level_0')
            out.index.name = ""
            if self.intercept_name and ('Intercept' in out.index):
                out.rename(index={'Intercept':self.intercept_name},inplace=True)
            if self.nobs:
                out = out.append(obs)
            if self.rsq:
                out = out.append(rsqs)

                
        elif self.orientation == 'horizontal':
            out = pd.concat(res,axis=0,ignore_index=True,sort=False).reset_index(drop=True)
            if self.nobs:
                obs = pd.Series([v for s in [[n,''] for n in list(obs)] for v in s],name='Obs')
                out = pd.concat([out,obs],axis=1)
            if self.rsq:
                rsqs = pd.Series([v for s in [[n,''] for n in list(rsqs)] for v in s],name='Rsq')
                out = pd.concat([out,rsqs],axis=1)
            if self.intercept_name and ('Intercept' in out.columns):
                out.rename(columns={'Intercept':self.intercept_name},inplace=True)
            if self.regnames != None:
                out.index = [v for s in [[n,''] for n in self.regnames] for v in s]

        self.out = out.fillna('')
        
    def render(self):
        self._assertions()
        self._form()
        return self.out
                
    def to_latex(self):
        self._assertions()
        self._form()
        return self.out.to_latex()
        
    def set_reg_names(self,vars):
        if type(vars) == list:
            assert len(vars)==self.num_reg, "Length of regnames input must equal regformulas length"
            self.regnames = vars        
        elif type(vars) == str:
            if vars == 'num-brackets':
                self.regnames = ['['+str(x)+']' for x in range(1,self.num_reg+1)]
            else:
                raise InputError('Invalid input for variable names.')
        
    def _clean_variables(self,ret):
        vars = ret.index
        ret.index = [x.replace('C(','').replace(')[T.','[').replace(']:',']*').replace('np.','').replace('[T.','[')
                          .replace(')[','[').replace(':','*') for x in vars]
        return ret
            
    def _stack(self,r):
        
        b_ = 'bstar' if (self.sig == 'coeff') else 'b'
        stat_ = self.stat+'star' if (self.sig == 'stat') else self.stat

        b = r.params.apply(lambda x: '{:{fmt}}'.format(x,fmt=self.bfmt))
        se = r.bse.apply(lambda x: '({:{fmt}})'.format(x,fmt=self.sfmt))
        tstat = r.tvalues.apply(lambda x: '({:{fmt}})'.format(x,fmt=self.sfmt))
        pvalues = r.pvalues.apply(lambda x: '({:{fmt}})'.format(x,fmt=self.sfmt))        
        star = pd.cut(r.pvalues,self.sigbins,labels=self.siglabels).astype(str)
        bstar = b + star
        tstatstar = tstat + star
        pvaluesstar = pvalues + star
        sestar = se + star 
        ret =   (pd.concat([b,se,tstat,pvalues,bstar,tstatstar,pvaluesstar,sestar],axis=1)  
                .rename(columns={0:'b',1:'se',2:'tstat',3:'pvalues',4:'bstar',5:'tstatstar',
                6:'pvaluesstar',7:'sestar'})[[b_,stat_]])
        ret = self._clean_variables(ret)
        
        if self.orientation == 'vertical':
            ret = ret.stack()
        elif self.orientation == 'horizontal':
            ret = ret.T
        return ret

    def get_coefficients(self,orientation='vertical'):
        assert orientation=='vertical' or orientation=='horizontal', 'Invalid input for orientation'
        d = pd.DataFrame([r.params for r in self.regressions])
        if self.regnames:
            d.index = self.regnames
        if orientation=='horizontal':
            d = d.T
        return d.T
        
    def get_se(self,orientation='vertical'):
        assert orientation=='vertical' or orientation=='horizontal', 'Invalid input for orientation'
        d = pd.DataFrame([r.bse for r in self.regressions])
        if self.regnames:
            d.index = self.regnames
        if orientation=='horizontal':
            d = d.T
        return d.T
        
    def get_tstats(self,orientation='vertical'):
        assert orientation=='vertical' or orientation=='horizontal', 'Invalid input for orientation'
        d = pd.DataFrame([r.tvalues for r in self.regressions])
        if self.regnames:
            d.index = self.regnames
        if orientation=='horizontal':
            d = d.T
        return d.T
        
    def get_pvalues(self,orientation='vertical'):
        assert orientation=='vertical' or orientation=='horizontal', 'Invalid input for orientation'
        d = pd.DataFrame([r.pvalues for r in self.regressions])
        if self.regnames:
            d.index = self.regnames
        if orientation=='horizontal':
            d = d.T
        return d.T
        