import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

class Regtable:
    """
    
    Class built to house multiple regressions and build corresponding tabular output.

    
    Parameters
    ----------

    data: Pandas.core.frame.DataFrame
        This DataFrame object should contain all variables which will be used in regressions.
    regformulas: list(str)
        A list of regression formulas to be run using the data parameter object. Formulas 
        must comply with the formatting of statsmodels.formula.api linear regression functions.
    regnames: list, optional 
        A list of names or labels corresponding to the regformulas provided. The length of 
        regnames must match the length of regformulas. Defaults to the dependent variable names
        for each regression.
    regtype: {'OLS','GLS','WLS','GLSAR'}, default = 'OLS'
        Specify the type of regression to be used. 
    orientation: {'vertical','horizontal'}, default = 'vertical'
        Specify whether to have the independent variables on the vertical or horizontal axis of 
        the table. 
    stat: {'tstat','se','pvalues'}, default = 'tstat'
        Specify which statistic to report in table. 
    sig: {False, 'stat','coeff'}, default = False
        Whether to display statistical significance with '*' or other symbols in table output.
        Defaults to False, which does not display significance in tables. If statistical significance 
        is desired to be shown with regression coefficients, set sig = 'coeff'. If statistical 
        significance is desired to be shown with the reported statistic, set sig = 'stat'.
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
    regformulas: list(str)
        A list of regression formulas to be run using the data parameter object. Formulas 
        must comply with the formatting of statsmodels.formula.api linear regression functions.
    regnames: list(str), optional 
        A list of names or labels corresponding to the regformulas provided. The length of 
        regnames must match the length of regformulas. Defaults to the dependent variable names
        for each regression.
    regtype: {'OLS','GLS','WLS','GLSAR'}, default = 'OLS'
        Specify the type of regression to be used. 
    orientation: {'vertical','horizontal'}, default = 'vertical'
        Specify whether to have the independent variables on the vertical or horizontal axis of 
        the table. 
    stat: {'tstat','se','pvalues'}, default = 'tstat'
        Specify which statistic to report in table. 
    sig: {False, 'stat','coeff'}, default = False
        Whether to display statistical significance with '*' or other symbols in table output.
        Defaults to False, which does not display significance in tables. If statistical significance 
        is desired to be shown with regression coefficients, set sig = 'coeff'. If statistical 
        significance is desired to be shown with the reported statistic, set sig = 'stat'.
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
        Resets the names of the regressions (regnames). However, it can also take string arguments which set
        the names according to a particular convention. The only currently available such option is 'num-brackets'
        which sets [1],[2],... as names for the regressions.
    

    """
    
    def __init__(self, data, regformulas, regnames=None, regtype='OLS', orientation='vertical',
                 stat='tstat',sig=False,sig_lbls={'***':0.01,'**':0.05},bfmt='.3f',sfmt='.2f',rsq=True,
                 nobs=True,intercept_name=None):
        
        self.orientation = orientation
        self.regnames = regnames
        self.regformulas = regformulas
        self._lag_regformulas = regformulas
        self.num_reg = len(regformulas)
        self.stat = stat
        self.intercept_name = intercept_name
        self.bfmt = bfmt
        self.sfmt = sfmt 
        self.rsq = rsq
        self.nobs = nobs
        self.data=data
        self.sig = sig
        self.sig_lbls = sig_lbls
        self.out = None
        self.regtype = regtype
        self._lag_regtype = regtype
        self._gen_models()
        
    def _gen_models(self):
        if self.regtype == 'OLS':
            self.lm = [smf.ols(f,data=self.data).fit() for f in self.regformulas]
        elif self.regtype == 'WLS':
            self.lm = [smf.wls(f,data=self.data).fit() for f in self.regformulas]
        elif self.regtype == 'GLS':
            self.lm = [smf.gls(f,data=self.data).fit() for f in self.regformulas]
        elif self.regtype == 'GLSAR':
            self.lm = [smf.glsar(f,data=self.data).fit() for f in self.regformulas]
    
    def _form(self):
        
        if (self.regtype != self._lag_regtype) or (self.regformulas != self._lag_regformulas):
            self._gen_models()
            self._lag_regtype = self.regtype
            self._lag_regformulas = self.regformulas
        
        sig_ = sorted(self.sig_lbls.items(), key=lambda kv:kv[1])
        self.sigbins = [x[1] for x in sig_]
        self.sigbins.append(1.01)
        self.sigbins.insert(0,-0.01)
        self.siglabels = [x[0] for x in sig_]
        self.siglabels.append('')

        if self.regnames == None:
            cols = [x.model.endog_names for x in self.lm]    
        else:
            cols = self.regnames
        res = [self._stack(r) for r in self.lm]
        
        obs = pd.Series([int(r.nobs) for r in self.lm],cols,name='Obs')
        rsqs = pd.Series(['{:{fmt}}'.format(r.rsquared,fmt=self.sfmt) for r in self.lm],cols,
                        name='Rsq')

        if self.orientation == 'vertical':
            out = pd.concat(res,axis=1,keys=cols).reset_index()
            out.loc[out['level_1'] == self.stat,'level_0'] = ''
            out.loc[out['level_1'] == (self.stat+'star'),'level_0'] = ''
            out = out.drop('level_1',axis=1).set_index('level_0')
            out.index.name = ""
            if self.intercept_name:
                out.rename(rows={'Intercept':self.intercept_name},inplace=True)
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
            if self.intercept_name:
                out.rename(columns={'Intercept':self.intercept_name},inplace=True)
            if self.regnames != None:
                out.index = [v for s in [[n,''] for n in self.regnames] for v in s]

        self.out = out.fillna('')
        
    def render(self):
        """
        Documentation to come
        """
        self._form()
        return self.out
                
    def to_latex(self):
        """
        Documentation to come
        """
        if type(self.out) != pd.core.frame.DataFrame:
            self._form()
        return self.out.to_latex()
        
    def set_reg_names(self,vars):
        """
        Documentation to come
        """
        if type(vars) == list:
            # TODO: Error checking for correct list length
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

