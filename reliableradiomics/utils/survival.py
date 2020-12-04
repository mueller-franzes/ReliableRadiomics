from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import warnings
import numpy as np
import logging
import pandas as pd
logger = logging.getLogger(__name__)

def time2label(x):
    if x<=360: # < 1 years 
        return 1
    elif x<=720: # < 2 years 
        return 2
    elif x<=1080: # < 3 years 
        return 3
    else:  # > 3 years 
        return 4

def cluster_surv_time(surv_time, surv_event):
    return [ time2label(time) if observed else -time2label(time)-1  for time, observed in zip(surv_time, surv_event) ]


def skf_split(x, y, n_splits=5, shuffle=True, random_state=0 ):
    surv_time, surv_event = y['time'], y['event']
    clusters = cluster_surv_time(surv_time, surv_event)
    
    if n_splits > 0 and n_splits < 1 :
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=n_splits, random_state=random_state)
    else:
        splitter =  StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"The least populated class in y has only.*")
        return [a for a in splitter.split(x, y=clusters)]
 

def stratified_cross_val(estimator, x, y, n_splits=5, shuffle=True, random_state=0):
    scores = [] 
    
    for idx_train, idx_test in skf_split(x, y, n_splits, shuffle, random_state):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                estimator.fit(x[idx_train], y[idx_train])
                scores.append(estimator.score(x[idx_test], y[idx_test]))
            except Warning as w:
                logger.warning("Warning: {} , excluding feature".format(w))
                scores.append(np.NaN)
            except:
                logger.warning("Predictor does not fit, excluding feature")
                scores.append(np.NaN)
    return np.asarray(scores)


def safe_val(estimator, x,y):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            estimator.fit(x, y)
            return estimator.score(x, y)
        except Warning as w:
            # logger.warning("Warning: {}, excluding feature".format(w))
            return np.NaN
        except:
            logger.warning("Predictor does not fit, excluding feature")
            return np.NaN 



def norm_rad_features(x):
    # x [Samples, Raters]
    return (x-np.mean(x, axis=0))/np.std(x, ddof=1, axis=0)
    # return x



def load_survival(dataset, path_surv_data):
    if dataset == "brats":
        df = pd.read_csv(path_surv_data)
        uid = df.loc[:,'BraTS19ID'].str.replace('_', '-')+"_0" # PatientID_CaseIdx
        survival_time =  df.loc[:,'Survival'].str.extract('(\d+)', expand=False) # extract time eg. 'ALIVE (1029 days later)' 
        event_observed = df.loc[:,'Survival'].str.isnumeric() & ~df.loc[:,'Survival'].isna() # here a number indicates an observed death status 
        age = df.loc[:, 'Age']
        data = {'uid':uid.to_numpy(str), 'time': survival_time.to_numpy(np.float32), 'event': event_observed.to_numpy(bool), 
                'age':age.to_numpy(np.float32) }
       
        return np.asarray([val for val in zip(*data.values()) if not np.isnan(val[1]) ], 
                           dtype=[(key, val.dtype ) for (key, val) in data.items()]  )
    
    elif dataset == "nsclc":
        df = pd.read_csv(path_surv_data)
        uid = df.loc[:,'PatientID'].str.split('LUNG1-').str[1] # PatientID
        survival_time =  df.loc[:, 'Survival.time']
        event_observed = df.loc[:, 'deadstatus.event']
        age = df.loc[:, 'age']
        sex = df.loc[:, 'gender']
        data = {'uid':uid.to_numpy(str), 'time': survival_time.to_numpy(np.int32), 'event': event_observed.to_numpy(bool), 
                'age':age.to_numpy(np.float32), 'sex':sex.to_numpy(str)  }
       
        return np.asarray([val for val in zip(*data.values())], 
                           dtype=[(item[0], item[1].dtype) for item in data.items()]  )