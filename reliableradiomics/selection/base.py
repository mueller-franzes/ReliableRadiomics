
from sklearn.model_selection import KFold,StratifiedKFold
import numpy as np
import logging
import warnings
from multiprocessing import Pool, get_context, cpu_count

from reliableradiomics.utils.survival import stratified_cross_val, safe_val


logger = logging.getLogger(__name__)

    

class SFS():
    def __init__(self, predictor, k, max_sel_feat=None, n_splits=None):
        self.predictor = predictor 
        self.k = k 
        self.max_sel_feat = max_sel_feat
        self.mask = None 
        self.n_splits = n_splits
        
        

    def fit(self, x,y):
        n_samples, n_feat = x.shape
        n_sel_feat = n_feat if self.k == "auto" else  self.k 
        max_sel_feat = n_feat if self.max_sel_feat is None else self.max_sel_feat 
        
        if n_sel_feat > n_feat:
            raise ValueError("Should select more features than exists")

        self.y = y   
        self.x = x 

        selected_feat, i_sel_feat, total_best_score = [], 0, 0
        while (i_sel_feat < n_sel_feat) and (i_sel_feat < max_sel_feat):

            # Compute scores for subsets with not yet selected features              
            temp_selected_feat = [[*selected_feat, i_feat] for i_feat in range(n_feat) if i_feat not in selected_feat]
            processes = min(cpu_count(), len(temp_selected_feat))
            with get_context().Pool(processes=processes) as pool:
                scores = pool.map(self._score, temp_selected_feat, chunksize=len(temp_selected_feat) // processes)

            # Pick best performaning  
            idx_best_score = np.nanargmax(scores)
            subset_best_score = scores[idx_best_score]

            # Check if total performance increases 
            if total_best_score<subset_best_score:
                total_best_score = subset_best_score
            elif total_best_score>=subset_best_score and self.k == "auto":
                logger.info("Performance decreases {} => {}".format(total_best_score, subset_best_score))
                break 

            # Add best feature to subset 
            selected_feat = temp_selected_feat[idx_best_score]
            i_sel_feat += 1
            logger.debug("Adding feature {}, total {}, Performance {}".format(idx_best_score, i_sel_feat, subset_best_score))

        self.mask = np.zeros(n_feat, dtype=bool)
        self.mask[selected_feat] = True 

        return self
    
    def transform(self, x):
        if self.mask is not None:
            return x[:,self.mask]
        else:
            raise ValueError("First exectue fit().")
    
    def fit_transform(self, x,y):
        return self.fit(x,y).transform(x)

    def _score(self, selected_feat):
        x_temp = self.x[:,selected_feat]

        # Without Train-Test split or with Cross Validation 
        if self.n_splits is None:   
            subset_score = safe_val(self.predictor, x_temp, self.y)
        else:
            subset_score = np.mean(stratified_cross_val(self.predictor, x_temp, self.y, n_splits=self.n_splits))

        return subset_score




