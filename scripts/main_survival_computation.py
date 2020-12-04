
from pathlib import Path
import numpy as np 
import json
import sys 
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator

from reliableradiomics.selection  import SFS
from reliableradiomics.utils.radiomics import load_feature_evaluation, load_features
from reliableradiomics.metrics.statistics import logrank_test
from reliableradiomics.utils.survival import skf_split, stratified_cross_val, norm_rad_features, load_survival
from reliableradiomics.utils.helpers import dict_append, dict_insert
from reliableradiomics.utils.logutils import NumpyEncoder


import logging
logger = logging.getLogger(__name__)





if __name__ == "__main__":

    # ------------------ Load Data -------------------------------
    dataset = 'nsclc' 
    
    path_surv = Path.cwd()/('data/survival/'+dataset+'/survival.csv') 
    path_features = Path.cwd()/('results/features/'+dataset+'/radiomics.csv') 
    path_feature_stability = Path.cwd()/('results/reliability/'+dataset+'/radiomic_scores.json') 
    path_feature_stability_multi = Path.cwd()/('results/reliability/all/radiomic_scores.json') 

    path_out =  Path.cwd()/('results/survival'+dataset)
    path_out.mkdir(parents=True, exist_ok=True)

    # Logging 
    s_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(path_out/'survial_log.log', 'w')
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[s_handler, f_handler])


    # Load Feature Stability Values 
    rad_metric_val = load_feature_evaluation(path_feature_stability)['Pred']['ICC(1)']
    sorted_rad_stab = {key: rank for rank, (key,val) in enumerate(sorted(rad_metric_val.items(), reverse=False, key=lambda item: item[1][1]))}
    feature_class_names = set([name.split('_')[1] for name in rad_metric_val.keys()])

    # Load Feature Stability (Multiple Datasets)
    rad_metric_val_mul = load_feature_evaluation(path_feature_stability_multi)['Pred']['ICC(1)']
    sorted_rad_stab_mul = {key: rank for rank, (key,val) in enumerate(sorted(rad_metric_val_mul.items(), reverse=False, key=lambda item: np.median(np.asarray(item[1])[:,1]) ))} 

    # Load Survival Data 
    survial_data = load_survival(dataset, path_surv)
    surv_uids = survial_data['uid'] # PatientID_(CaseIdx)

    # Load Feature Values 
    feat, feat_uids, sel_col = load_features(path_features) # feat_uids [Patients, Raters ] with PatientID_(CaseIdx)_LesionIdx_SliceIdx_RaterIdx
    num_raters = {anno:len(sel_col[anno]) for anno in feat.keys()}
    feat_uids = [fuid.rsplit('_',3)[0] for fuid in feat_uids['GT'][:,0]] # PatientID_(CaseIdx)

    # Find cases that are availabel across datasets  
    uids, feat_ind, surv_ind = np.intersect1d(feat_uids, surv_uids, return_indices=True)

    # Remove unavailable cases and norm radiomic features 
    survial_data = survial_data[surv_ind]
    feat = {anno:{feat_name:norm_rad_features(feat[anno][feat_name][feat_ind]) for feat_name in feat[anno].keys()} for anno in feat.keys()}




    # -------------------  Compute C-Index for Individual features -------------------- 
    results = {}

    logger.info("Start C-Index computation for each feature.")
    for anno in feat.keys():
        n_feat = len(feat[anno])
        for k_feat, f_name in enumerate(feat[anno].keys()):
            rater_dict = {'rank':[], 'icc':[], 'c-index':[]} 
            for rater in range(num_raters[anno]):
                x = feat[anno][f_name][:,rater][:,None]  # (n_samples, m_features)
                y = survial_data[['event', 'time']]

                # Model 
                predictor  = CoxPHSurvivalAnalysis(alpha=0, n_iter=1e9)

                # Evaluation 
                c_indexes = stratified_cross_val(predictor, x, y, n_splits=5).tolist()
                rater_dict['c-index'].append(c_indexes)
                
                
                # Option 1: Stability ranking based on this dataset  
                rater_dict['rank'].append(sorted_rad_stab[f_name])
                rater_dict['icc'].append(rad_metric_val[f_name])

                # Option 2: Stability ranking based on all datasets
                # rater_dict['rank'].append(sorted_rad_stab_mul[f_name])
                # rater_dict['icc'].append(rad_metric_val_mul[f_name])

                logger.info("{} Rater {} - {}/{} Feature {}: C-Index-Mean: {} ".format(anno, rater, k_feat, n_feat, f_name, np.mean(c_indexes) ))
                
            dict_insert(results, rater_dict, anno, f_name)


    with open(path_out/'survival_ind.json', 'w') as f:
        json.dump(results, f)





    # ---------------------- Compute C-Index/KM for Signature --------------------------
    results = {}

    # Train-Test split
    splits = [{'train':split[0], 'test':split[1]} for split in skf_split(survial_data, survial_data, n_splits=5)]


    # --------------------  1.) Feature Selection 
    for mode in [  'High ICC',  'All ICC', 'Low ICC',   ]: 
        logger.info("-----Start feature selection for mode {} -----".format(mode))

        if mode == "All ICC":
            feat_gt_subset = feat['GT'] 
        elif mode =="High ICC":
            high_stab_feat = [feat_name for feat_name, icc_val in rad_metric_val_mul.items() if np.median(np.asarray(icc_val)[:,1])>=0.99]
            feat_gt_subset = { feat_name:feat['GT'][feat_name] for feat_name in high_stab_feat}
        elif mode =="Low ICC":
            low_stab_feat = [feat_name for feat_name, icc_val in rad_metric_val_mul.items() if np.median(np.asarray(icc_val)[:,1])<=0.75]
            feat_gt_subset = { feat_name:feat['GT'][feat_name] for feat_name in low_stab_feat}
        else:
            raise ValueError()

        logger.info("{} features in ICC-subset {}".format(len(feat_gt_subset), mode))
        feat_gt_subset_names = np.asarray(list(feat_gt_subset.keys())) 
        feat_gt_subset_val = np.asarray(list(feat_gt_subset.values()))[:,:,0] # [n_features, n_samples, n_raters=0]
        feat_gt_subset_val = np.swapaxes(feat_gt_subset_val, 0,1) # (n_samples, n_features)


        selected_features = []
        for n_split, split in enumerate(splits):
            # Split
            logger.debug("Split {}".format(n_split+1)) 
            x_train_gt =  feat_gt_subset_val[split['train']]
            y_train_gt =  survial_data[split['train']]

            # Model 
            predictor  = CoxPHSurvivalAnalysis(alpha=0, n_iter=1e9)
            
            # Sequential Based Selection  
            selector = SFS(predictor, k=3)
            selector.fit(x_train_gt, y_train_gt[['event', 'time']])
            mask = selector.mask 

            # Safe selected features  
            selected_features.append(feat_gt_subset_names[mask])
            logger.info("Selected features in mode {} split{}: {}".format(mode, n_split+1, selected_features[-1]))


        dict_append(results, selected_features, 'selected_features', mode)


        
        
        # ----------------------- C-Index computation ----------------------
        logger.info("--- Start C-Index Computation in mode {} ----".format(mode))
        for anno in feat.keys():
            logger.info("Annotator: {}".format(anno))
            for rater in range(num_raters[anno]):
                logger.info("Rater: {}".format(rater))
                c_indexes, high_risk_masks, y_tests = [], [], []
                for n_split, split in enumerate(splits):
                    logger.info("Split: {}".format(n_split+1))

                    x = np.asarray([feat[anno][f_name][:,rater] for f_name in selected_features[n_split]])
                    x = np.swapaxes(x, 0,1) # (n_samples, n_features)
                    
                    x_train, x_test = x[split['train']],  x[split['test']]
                    y_train, y_test = survial_data[split['train']], survial_data[split['test']]

                    # Model 
                    predictor  = CoxPHSurvivalAnalysis(alpha=0, n_iter=1e9)
                
                    try:
                        predictor.fit(x_train, y_train[['event', 'time']])
                        c_indexes.append(predictor.score(x_test, y_test[['event', 'time']]))
                        risk_score_train = predictor.predict(x_train)
                        risk_score = predictor.predict(x_test)
                        high_risk_masks.append(risk_score>np.median(risk_score_train))
                        y_tests.append(y_test)
                    except Exception as e:
                        logger.warning("Error {}".format(str(e)))
                        c_indexes.append(np.NaN)
                

                # ----------------------- Kaplan-Meier --------------------------------
                high_risk_mask = np.concatenate(high_risk_masks)
                
                y_tests = np.concatenate(y_tests)
                y_high_risk, y_low_risk = y_tests[high_risk_mask], y_tests[~high_risk_mask]

                km_high_time, km_high_prob = kaplan_meier_estimator(y_high_risk['event'], y_high_risk['time'])
                km_low_time, km_low_prob = kaplan_meier_estimator(y_low_risk['event'], y_low_risk['time'])
                km_ests = [[km_high_time.tolist(), km_high_prob.tolist()], [km_low_time.tolist(), km_low_prob.tolist()]]
                
                p_vals = logrank_test(y_tests[['event', 'time']], high_risk_mask)

                dict_append(results, c_indexes, anno, 'c-index', mode)
                dict_append(results, p_vals, anno, 'kaplan-meier_p-value', mode)
                dict_append(results, km_ests, anno, 'kaplan-meier_estimate', mode)

        
    # Add class names for simplification 
    results['feature_class_names'] = list(feature_class_names)



    # Save results 
    with open(path_out/'survival_sig.json', 'w') as f:
        json.dump(results, f, cls=NumpyEncoder)



