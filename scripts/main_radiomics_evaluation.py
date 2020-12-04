import numpy as np 
import json
import matplotlib.pyplot as plt 
from pathlib import Path 

import reliableradiomics.metrics.statistics as stats
from reliableradiomics.utils.radiomics import load_features
from reliableradiomics.utils.logutils import NumpyEncoder 



def plot_ds(dataset, only_plot=True):
    # ----------- Settings ---------
    path_features = Path.cwd()/('results/features/'+dataset+'/radiomics.csv')   
    path_output = Path.cwd()/('results/reliability/'+dataset)

    path_output.mkdir(parents=True, exist_ok=True)
    
    anno_names = ['GT', 'Pred']
    real_names = {'GT': 'Experts', 'Pred': 'PHiSeg'}
    icc_types = ["ICC(1)"] # ["ICC(1)", "ICC(A,1)", "ICC(C,1)" ]



    # ----------------- Load Data -----------------------------------
    # Load Radiomics Features
    feature_values, _, sel_cols = load_features(path_features)
    mul_raters = {anno:len(sel_cols[anno])>1 for anno in anno_names}
    
    # Get class/group  names like "firstorder" 
    feature_class_names = set([name.split('_')[1] for name in feature_values['GT'].keys()])
    
    feature_class_names_select = ['firstorder', 'shape2D', 'glcm', 'glszm', 'gldm', 'glrlm', 'ngtdm' ]
    feature_class_names = [fc_name for fc_name in feature_class_names_select if fc_name in feature_class_names]
    feature_class_values = {}
    for anno in feature_values.keys():
        feature_class_values[anno] = {}
        for feature_class_name in feature_class_names:
            feature_class_values[anno][feature_class_name] = {}
            for f_name in feature_values[anno].keys():
                if f_name.split('_')[1] == feature_class_name:
                    feature_class_values[anno][feature_class_name][f_name] = feature_values[anno][f_name]



    # Load data or compute 
    if only_plot:
        with open(path_output/'radiomic_scores.json', 'r') as f:
            rad_metrics = json.load(f)
    else:
        rad_metrics = {anno:{} for anno in anno_names}   
        
        # ICC  
        for anno in anno_names:
            if not mul_raters[anno]:
                continue
            for feat_name, val in feature_values[anno].items():
                icc = stats.ICC(val, ci_method="bootstrap")
                for metric in icc_types:
                    if  metric not in rad_metrics[anno]:
                        rad_metrics[anno][metric] = {}
                    rad_metrics[anno][metric][feat_name] = icc(metric)
        
                    
        # Write raw data to file 
        with open(path_output/'radiomic_scores.json', 'w') as f:
            json.dump(rad_metrics, f, cls=NumpyEncoder)

    

    # Group radiomics results 
    metric_class_values = {anno:{} for anno in anno_names}  
    for anno in anno_names:
        if not mul_raters[anno]:
            continue
        for metric in rad_metrics[anno].keys():        
            metric_class_values[anno][metric] = {}
            for feat_class_name in feature_class_names:
                metric_class_values[anno][metric][feat_class_name] = {feat_name:feat_val for feat_name, feat_val in rad_metrics[anno][metric].items() if feat_name.split('_')[1] == feat_class_name}


    # --------------------- Visualization -------------------
    # ICC - separated by feature groups 
    for icc_type in icc_types : 
        num_feat = [len(val.keys()) for val in metric_class_values['Pred'][icc_type].values()]
        fig, ax = plt.subplots(ncols=1, nrows=len(feature_class_names), figsize=(6,18) , sharey=False, sharex=True,
                               gridspec_kw={'height_ratios': num_feat}) 
        for n, fcn in enumerate(feature_class_names):
            pred_val = np.asarray(list(metric_class_values['Pred'][icc_type][fcn].values()))
            feat_names = np.asarray([name for name in metric_class_values['Pred'][icc_type][fcn].keys()])
            sort_idx = np.argsort(-pred_val[:,1])
            
            axis = ax[n]
            axis.grid(True)
            axis.set_axisbelow(True)
            axis.set_title(icc_type+" "+fcn, fontdict={'fontsize': 8, 'fontweight': 'bold'})

            offset = 0
            if mul_raters['GT']: 
                gt_val = np.asarray(list(metric_class_values['GT'][icc_type][fcn].values()))
                offset = 0.1
                y = np.arange(len(feat_names))-offset
                axis.scatter(y=y, x=gt_val[sort_idx,1], marker='D', s=2**2, c='r', label=real_names['GT']) 
                for x_l,y_l in zip(gt_val[sort_idx], y) :
                    axis.plot(x_l[(0,2),], [y_l,y_l], c='r', marker="|",  markersize=2,  linewidth=0.5, label=None) 
        

            y = np.arange(len(feat_names))+offset
            axis.scatter(y=y, x=pred_val[sort_idx,1], marker='o', s=2**2, c='b', label=real_names['Pred'])
            for x_l,y_l in zip(pred_val[sort_idx], y) :
                axis.plot(x_l[(0,2),], [y_l,y_l], c='b',  marker="|", markersize=2, linewidth=0.5, label=None) 


            axis.set_yticks(range(len(feat_names)))
            axis.set_yticklabels(feat_names[sort_idx], fontdict={'fontsize': 3, 'fontweight': 'normal'}) 
            axis.set_ylim([-1, len(feat_names)])
            axis.spines['right'].set_visible(False)
            axis.spines['top'].set_visible(False)       
            if n == len(feature_class_names)-1:
                axis.legend(loc='lower left')
            else:
                axis.spines['bottom'].set_visible(False)  
                axis.tick_params(bottom=False)
            
            if "ICC" in icc_type:
                axis.set_xticks(np.arange(0,1.1, 0.1))
                axis.set_xlim([0,1.05])
            
        fig.tight_layout()
        fig.savefig(path_output/(icc_type+'.png'), dpi=300)



    # ICC - not separated by feature groups 
    for icc_type in icc_types  :
        fig, axis = plt.subplots(figsize=(6,18)) 
        
        pred_val = np.asarray(list(rad_metrics['Pred'][icc_type].values()))
        feat_names = np.asarray([name for name in rad_metrics['Pred'][icc_type].keys()])
        sort_idx = np.argsort(-pred_val[:,1])
        
       
        axis.grid(True)
        axis.set_axisbelow(True)
        axis.set_title(icc_type, fontdict={'fontsize': 8, 'fontweight': 'bold'})

        offset = 0
        if mul_raters['GT']: 
            gt_val = np.asarray(list(rad_metrics['GT'][icc_type].values()))
            offset = 0.1
            y = np.arange(len(feat_names))-offset
            axis.scatter(y=y, x=gt_val[sort_idx,1], marker='D', s=2**2, c='r', label=real_names['GT']) 
            for x_l,y_l in zip(gt_val[sort_idx], y) :
                axis.plot(x_l[(0,2),], [y_l,y_l], c='r', marker="|",  markersize=2,  linewidth=0.5, label=None) 
        

        y = np.arange(len(feat_names))+offset
        axis.scatter(y=y, x=pred_val[sort_idx,1], marker='o', s=2**2, c='b', label=real_names['Pred'])
        for x_l,y_l in zip(pred_val[sort_idx], y) :
            axis.plot(x_l[(0,2),], [y_l,y_l], c='b',  marker="|", markersize=2, linewidth=0.5, label=None) 

        axis.set_yticks(range(len(feat_names)))
        axis.set_yticklabels(feat_names[sort_idx], fontdict={'fontsize': 3, 'fontweight': 'normal'}) 
        axis.set_ylim([-1, len(feat_names)])
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)       
        axis.legend(loc='lower left')
      
        if "ICC" in icc_type:
            axis.set_xticks(np.arange(0,1.1, 0.1))
            axis.set_xlim([0,1.05])
            
        fig.tight_layout()
        fig.savefig(path_output/(icc_type+'_not_grouped.png'), dpi=300)



 


if __name__ == "__main__":
    dataset = 'nsclc'
    plot_ds(dataset, only_plot=False)
  