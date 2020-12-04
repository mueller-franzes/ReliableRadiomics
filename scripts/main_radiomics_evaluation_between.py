
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.transforms
from pathlib import Path 
import json


if __name__ == "__main__":
    # ----------- Settings ---------
    datasets = {
                # 'LIDC':  Path.cwd()/('results/reliability/lidc/radiomic_scores.json'),
                # 'KITS':  Path.cwd()/('results/reliability/kits/radiomic_scores.json'),
                # 'LITS': Path.cwd()/('results/reliability/lits/radiomic_scores.json'),
                'NSCLC':  Path.cwd()/('results/reliability/nsclc/radiomic_scores.json'),
                # 'BRATS': Path.cwd()/('results/reliability/brats/radiomic_scores.json')
                } 

    ds_colors =  {'LIDC': 'g', 'KITS':'r', 'LITS':'b', 'NSCLC':'m', 'BRATS':'c'}

    path_output = Path.cwd()/('results/reliability/all') 
    path_output.mkdir(parents=True, exist_ok=True)
    
    icc_types = ["ICC(1)" ]    # ["ICC(1)", "ICC(A,1)", "ICC(C,1)", "F" ]   
    anno_names = ['GT', 'Pred']
    use_anno = "Pred" # Which annotator should be used for stability evaluation [GT, Pred] 



    # ----------------- Load Data -----------------------------------
    rad_metrics = {}
    metric_class_values = {}

    for dataset, data_path in datasets.items():

        with open(data_path, 'r') as f:
            rad_metrics[dataset]  = json.load(f)
        
      
        # Get class/group  names like "firstorder" 
        feature_names = rad_metrics[dataset][use_anno]['ICC(1)'].keys()
        feature_class_names = set([name.split('_')[1] for name in feature_names])
        feature_class_names_select = ['shape2D', 'firstorder',  'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm' ]
        feature_class_names = [fc_name for fc_name in feature_class_names_select if fc_name in feature_class_names]
        
  
        

        metric_class_values[dataset] = {use_anno: {}}
        for metric in ['ICC(1)']:        
            metric_class_values[dataset][use_anno][metric] = {}
            for feat_class_name in feature_class_names:
                metric_class_values[dataset][use_anno][metric][feat_class_name] = {feat_name:feat_val for feat_name, feat_val in rad_metrics[dataset][use_anno][metric].items() if feat_name.split('_')[1] == feat_class_name}

            

    with open(path_output/'radiomic_scores.json', 'w') as f:
        com_results = {use_anno:{}} 
        for metric in ['ICC(1)']:
            com_results[use_anno][metric] = {}
            for feat in feature_names:
                com_results[use_anno][metric][feat] = [rad_metrics[dataset][use_anno][metric][feat] for dataset in datasets]
        json.dump(com_results, f)
    


    # --------------------- Visualization -------------------
    
    # ICC - separated by feature groups 
    for icc_type in icc_types:
        ds_name = list(datasets.keys())[0]
        num_feat = [len(val.keys()) for val in metric_class_values[ds_name][use_anno][icc_type].values()]
        
        fig, ax = plt.subplots(ncols=1, nrows=len(feature_class_names), figsize=(6,18) , sharey=False, sharex=True,
                               gridspec_kw={'height_ratios': num_feat})

        for n, fcn in enumerate(feature_class_names):
            feat_names = np.asarray([name.rsplit('_',1)[1] for name in metric_class_values[ds_name][use_anno][icc_type][fcn].keys()])

            # Sort by Median 
            all_pred_iccs = np.stack([np.asarray(list(metric_class_values[ds][use_anno][icc_type][fcn].values()))[:,1] for ds in datasets.keys()])
            pred_val = np.median(all_pred_iccs, axis=0)
            sort_idx = np.argsort(-pred_val)
            
            axis = ax[n]
            axis.grid(True)
            axis.set_axisbelow(True)
            axis.set_title(icc_type+" "+fcn, fontdict={'fontsize': 8, 'fontweight': 'bold'})
            axis.set_yticks(range(len(feat_names)))
            axis.set_yticklabels(feat_names[sort_idx], fontdict={'fontsize': 3, 'fontweight': 'normal'}) 
            axis.set_ylim([-1, len(feat_names)])
            axis.spines['right'].set_visible(False)
            axis.spines['top'].set_visible(False)   

            # Plot Line 
            if len(datasets) >1:
                axis.plot(pred_val[sort_idx], np.arange(len(sort_idx)) , c='y', linewidth=1.5, label='Median') 

            for dataset in datasets.keys():
                pred_val = np.asarray(list(metric_class_values[dataset][use_anno][icc_type][fcn].values()))
                y = np.arange(len(feat_names))
                axis.scatter(y=y, x=pred_val[sort_idx,1], marker='o', c=ds_colors[dataset], s=2**2, label=''+dataset)
                # for x_l,y_l in zip(pred_val[sort_idx], y) :
                #     axis.plot(x_l[(0,2),], [y_l,y_l], c=ds_colors[dataset],  marker="|", markersize=2, linewidth=0.5, label=None) 
            
            
            if n == len(feature_class_names)-1:
                axis.legend(loc='lower left' , shadow=True, edgecolor='k')
            else:
                axis.spines['bottom'].set_visible(False)  
                axis.tick_params(bottom=False)
            
            if "ICC" in icc_type:
                axis.set_xticks(np.arange(0,1.1, 0.1))
                axis.set_xlim([0,1.05])

        fig.tight_layout()
        fig.savefig(path_output/(icc_type+'.png'), dpi=300)


        
    

    
    
    
    
    #  ICC - not separated by feature groups
    for icc_type in icc_types:
        fig, axis = plt.subplots(figsize=(6,18), dpi=300, gridspec_kw={ 'wspace':0,  'hspace':0}) 


        # Check if all features names are equal between different datasets 
        feat_names = {ds: np.asarray([name for name in rad_metrics[ds][use_anno][icc_type].keys()]) # name.rsplit('_',1)[1]
                        for ds in datasets.keys()}

        iter_feat_name = iter(feat_names.values())
        a = next(iter_feat_name)
        for n in range(len(datasets)-1):
            b = next(iter_feat_name)
            assert all(a==b), "Feature names differ"

        feat_names = a 
        
      
        # Sort by Median 
        pred_val = np.stack([np.asarray(list(rad_metrics[ds][use_anno][icc_type].values()))[:,1] for ds in datasets.keys()])
        pred_val = np.median(pred_val, axis=0)
        sort_idx = np.argsort(-pred_val)


        axis.grid(True)
        axis.set_axisbelow(True)
        axis.set_title(icc_type, fontdict={'fontsize': 8, 'fontweight': 'bold'})    
        for ds in datasets.keys(): 
            pred_val_ds = np.asarray(list(rad_metrics[ds][use_anno][icc_type].values())) 

            y = np.arange(len(feat_names))
            axis.scatter(y=y, x=pred_val_ds[sort_idx,1], marker='o', s=2**2, c=ds_colors[ds], label=ds)

            # 95 Confidence Interval 
            # for x_l,y_l in zip(pred_val_ds[sort_idx], y) :
            #     axis.plot(x_l[(0,2),], [y_l,y_l], c=ds_colors[ds],  marker="|", markersize=2, linewidth=0.5, label=None) 
        

        # Plot Line
        if len(datasets) >1:
            axis.plot(pred_val[sort_idx], np.arange(len(sort_idx)) , c='y', linewidth=1.5, label='Median') 
        
        axis.set_yticks(range(len(feat_names)))
        axis.set_yticklabels(feat_names[sort_idx], fontdict={'fontsize': 3, 'fontweight': 'normal'}) 
        axis.set_ylim([-1, len(feat_names)])
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)       
        axis.legend(loc='lower left' , shadow=True, edgecolor='k')
      
        if "ICC" in icc_type:
            axis.set_xticks(np.arange(0,1.1, 0.1))
            axis.set_xlim([0,1.05])

        fig.tight_layout()
        fig.savefig(path_output/(icc_type+'_not_grouped.png'), dpi=300)





    # ------------ Table -----------------------
    fig = plt.figure(figsize=(6,15), dpi=300) 
    
    columns = list(datasets.keys())
    rows = list(rad_metrics[columns[0]][use_anno]['ICC(1)'].keys()) 


    # Sort by Lower 95-CI
    pred_val = np.stack([np.asarray(list(rad_metrics[ds][use_anno]['ICC(1)'].values()))[:,0] for ds in datasets.keys()])
    pred_val = np.min(pred_val, axis=0)
    sort_idx = np.argsort(-pred_val)


    data  = np.asarray([list(rad_metrics[ds][use_anno]['ICC(1)'].values()) for ds in datasets.keys()])
    data = data[:,sort_idx]
    data = np.round(data,3) # [Datasets, Features, 3]
    data = np.moveaxis(data, 0,1) # [Features, Datasets, 3]

    data = [[ str(ds[1])+' ['+str(ds[0])+', '+str(ds[2])+']'  for ds in feat ] for feat in data]
 

    table = plt.table(cellText=data,
                      rowLabels=rows,
                      colLabels=columns,
                      loc='center') 

    
    plt.axis('off')
    plt.grid('off')

    # draw canvas once
    plt.gcf().canvas.draw()
    # get bounding box of table
    points = table.get_window_extent(plt.gcf()._cachedRenderer).get_points()
    # add 10 pixel spacing
    points[0,:] -= 10; points[1,:] += 10
    # get new bounding box in inches
    nbbox = matplotlib.transforms.Bbox.from_extents(points/plt.gcf().dpi)
    # save and clip by new bounding box
    plt.savefig(path_output/'table.pdf', bbox_inches=nbbox)
   

    
    