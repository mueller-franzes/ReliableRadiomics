import numpy as np 
import scipy.spatial.distance as distance
import scipy.stats as stats 
import sklearn.metrics 
from sksurv.compare import compare_survival


def ssum(a, axis=None, dtype=None, out=None, keepdims=np._NoValue,  initial=np._NoValue, where=np._NoValue, nan_policy='propagate'):
    # see https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    # see https://docs.scipy.org/doc/numpy/reference/generated/numpy.nansum.html
    if nan_policy == 'propagate' or nan_policy == 'raise':
        if nan_policy == 'raise' and np.isnan(a):
            raise ValueError("NaN value detected")
        return np.sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where)
    elif nan_policy == 'omit':
        return np.nansum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    else:
        assert False, "nan_policy: {} is not known".format(nan_policy)

def mean(a, axis=None, dtype=None, out=None, keepdims=np._NoValue, nan_policy='propagate'):
    # see https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html 
    # see https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanmean.html 
    if nan_policy == 'propagate' or nan_policy == 'raise':
        if nan_policy == 'raise' and np.isnan(a):
            raise ValueError("NaN value detected")
        return np.mean(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    elif nan_policy == 'omit':
        return np.nanmean(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    else:
        assert False, "nan_policy: {} is not known".format(nan_policy)
    

def ci(data, cl=0.95, axis=0, metric='mean', center=False):
    """Confidence Interval (CI)"""
    # Note: percentile vs  quantile vs quartile
    # Note: IQR = Inter Quartile Range  
    if metric == 'mean':
        n = data.shape[axis]
        mu = mean(data, axis=axis) 
        s = std(data, ddof=1, axis=axis)
        ppf = stats.t.ppf(q=(1 + cl) / 2, df=n-1) 
        if center:
            return [mu-ppf*s/np.sqrt(n), mu, mu+ppf*s/np.sqrt(n)]
        else:
            return [mu-ppf*s/np.sqrt(n), mu+ppf*s/np.sqrt(n)] 
    elif metric == "t-percentile":
        n = data.shape[axis]
        mu = mean(data, axis=axis) 
        s = std(data, ddof=1, axis=axis)
        ppf = stats.t.ppf(q=(1 + cl) / 2, df=n-1) 
        if center:
            return [mu-ppf*s, mu, mu+ppf*s]
        else:
            return [mu-ppf*s, mu+ppf*s] 
    elif metric == "rank-percentile":
        # e.q. cl=0.5 => intervall that covers 50% of the data centered by median 
        # Note: Only special case:  mean = median 
        # IQR = percentile[1] - percentile[0]
        q = np.percentile(data, q=((1-cl)/2*100,(1+cl)/2*100), axis=axis, out=None, overwrite_input=False, interpolation='linear', keepdims=False)
        if center:
            median = np.median(data, axis=axis, out=None, overwrite_input=False, keepdims=False)
            return [q[0], median, q[1]] 
        else:
            return q


class ICC(object):    
    def __init__(self, a, ci_method="none", **ci_kwargs):
        """ Intra-class Correlation Coefficient (ICC)
        
        Arguments:
            a {numpy.ndarray} -- Array of shape [Rows, Columns]. Each column contains one rater 
        """
        self.icc_dict = icc(a)
        if ci_method == "none":
            pass
        elif ci_method == "bootstrap":
            resample = ci_kwargs.pop("resample", 1000)
            n, _  = a.shape # targets/subjects, judges/raters
            icc_values =  np.asarray([list(icc(a[np.random.randint(0,n,n),:]).values()) for _ in range(resample)])
            icc_names = self.icc_dict.keys() # workaround to get names of availabe ICCs 
            boot_args={"metric":"rank-percentile", "center":True}
            boot_args.update(ci_kwargs)

            self.icc_dict= {icc_name:ci(icc_value, **boot_args) for icc_name, icc_value in zip(icc_names, icc_values.T)}
        else:
            raise ValueError("ci={} is not a valid argument".format(ci))
               

    def __call__(self, name):
        return self.icc_dict.get(name)



def icc(a):
    # see https://doi.org/10.1371/journal.pone.0219854 
    n , k  = a.shape # targets/subjects, judges/raters
    S = mean(a, axis=1) # Mean subject (row) 
    M = mean(a, axis=0) # Mean measurement (column) 
    x = mean(a) #  Total mean value

    SST = ssum(ssum((a-x)**2, axis=1)) # Sum of Squares, Total
    SSBS = ssum(k*(S-x)**2) # Sum of Squares Between Subjects
    SSBM = n*ssum((M-x)**2) # Sum of Squares Between Measurements
    SSE = SST - SSBS - SSBM # Sum of Squares, Error
    SSWS = SSBM + SSE # Sum of Squares Within Subjects
    SSWM = SSBS + SSE # Sum of Squares Within Measurements
    

    MSBS = SSBS/(n-1) # Mean Square Between Subjects 
    MSBM = SSBM/(k-1) # Mean Square Between Measurements
    MSWS = SSWS/(n*(k-1)) # Mean Square Within Subjects
    MSE = SSE/((n-1)*(k-1)) # Mean Square, Error)
  
    return {
        "ICC(1)": np.divide(MSBS-MSWS, MSBS+(k-1)*MSWS),
        "ICC(A,1)": np.divide(MSBS-MSE, MSBS+(k-1)*MSE+(k/n)*(MSBM-MSE)),
        "ICC(C,1)": np.divide(MSBS-MSE, MSBS+(k-1)*MSE),
        "F": np.divide(MSBS,MSE) 
    }



def dice(y_true, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None, zero_division='warn'):
    """ Dice similarity coefficient (DSC) or F1-Score """
    # WARNING: average='macro' as default 
    # see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    return sklearn.metrics.f1_score(y_true, y_pred, labels=labels, pos_label=pos_label, average=average, 
                                    sample_weight=sample_weight,  zero_division=zero_division)



def logrank_test(y, group_indicator, return_stats=False):    
    chisq, pval = compare_survival(y, group_indicator, return_stats=False)
    return pval
     




