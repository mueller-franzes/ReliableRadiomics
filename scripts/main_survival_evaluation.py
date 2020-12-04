
from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
import json


if __name__ == "__main__":

    # ------------------------ Load Data ------------------------------
    dataset = 'nsclc'
    path_features = Path.cwd()/('results/features/'+dataset+'/radiomics.csv') 
    path_survival =  Path.cwd()/('results/survival/'+dataset)
    
    path_out =  Path.cwd()/('results/survival/'+dataset)
    path_out.mkdir(parents=True, exist_ok=True)

    # Load survial statics 
    with open(path_survival/'survival_ind.json', 'r') as f:
        survival_ind = json.load(f)

    with open(path_survival/'survival_sig.json', 'r') as f:
        survival_sig = json.load(f)

    feature_class_names = survival_sig['feature_class_names']
    annos = ['Pred', 'GT']
    anno_names = {'GT': 'Expert', 'Pred': 'NN'}
    anno_marker = {'GT': 'X', 'Pred': 'o'}
    anno_marker_size = {'GT': 80, 'Pred':50}
    anno_marker_c = {'GT': 'r', 'Pred':'b'}
    fontdict={'fontsize': 10, 'fontweight': 'bold'}






    # --------------------- C-Index  Individual -----------------------
    # sorted by Rank 
    fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(12,9) ) 
    for anno in annos:
        x = np.asarray([data['rank'] for f_name, data in survival_ind[anno].items() ])  #[Samples, Raters]
        y = np.asarray([data['c-index'] for f_name, data in survival_ind[anno].items() ]) #[Samples, Raters, Splits]
        for n, yi in enumerate(np.moveaxis(y, -1, 0)): # Iterate over splits
            axis.scatter(x.flatten(), yi.flatten(), label=anno_names[anno] if n==0 else None, c=anno_marker_c[anno], alpha=0.5 if anno=='Pred' else 1, s=5, marker=anno_marker[anno])
    axis.legend(loc='lower left',  shadow=True, edgecolor='k')
    axis.set_axisbelow(True)
    axis.grid(True)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)  
    axis.set_title("{}".format(dataset.upper()), fontdict=fontdict)  
    axis.set_xlabel('ICC Rank (0= min stability)')
    axis.set_ylabel('C-Index')
    fig.tight_layout()
    fig.savefig(path_out/('c-index_ind_rank.png'), dpi=300)







    # ------------------------- C-Index  Signature ----------------------

    # C-Index - ScatterPlot
    fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(6,6) ) 
    for anno in annos:
        c_idx_sig = survival_sig[anno]["c-index"]
        signatures = list(c_idx_sig.keys()) 
        y = np.asarray([[rater for  rater in c_idx_sig[sig]]  for sig in signatures ]) # [Signature, Raters, Splits]
        y = np.reshape(y, (len(signatures), -1)) # [Signature, Splits*Raters]
        x = np.asarray([[n+1]*len(yi) for n, yi in enumerate(y)]) # [Signature, Splits*Raters*(SigIdx)]
        axis.scatter(x.flatten(), y.flatten(), marker=anno_marker[anno], s=anno_marker_size[anno], c=anno_marker_c[anno], label=anno_names[anno], alpha=0.25 if anno=="Pred" else 1)

    axis.legend(loc='best', shadow=True, edgecolor='k')
    axis.set_ylabel('C-Index', fontdict=fontdict)
    axis.set_xlabel('Signature', fontdict=fontdict)
    axis.set_title("{}".format(dataset.upper()), fontdict=fontdict)   
    axis.set_axisbelow(True)
    axis.grid(True)
    axis.set_xticks(list(range(1,len(signatures)+1)))
    axis.set_xticklabels(signatures)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)  
    fig.tight_layout()
    fig.savefig(path_out/('c-index_sig.png'), dpi=300)




    # C-Index - Barplots (Mean +- Std)
    fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(6,6) ) 
    data_box = [] 
    for anno in annos:
        for sig_name, sig_val in survival_sig[anno]["c-index"].items():
            for rater_val in sig_val:
                for split_val in rater_val:
                    data_box.append({'Signature': sig_name, 'C-index':split_val , 'Annotator':anno_names[anno]})

    df_box = pd.DataFrame(data_box)
    sns.barplot(ax=axis, x='Signature', y='C-index', hue='Annotator', data=df_box, estimator=np.mean, ci="sd")

    axis.legend(loc='lower left', shadow=True, edgecolor='k')
    axis.set_ylabel('C-Index', fontdict=fontdict)
    axis.set_ylim([0.45, 0.65])
    axis.set_xlabel('Signature', fontdict=fontdict)
    axis.set_title("{}".format(dataset.upper()), fontdict=fontdict)   
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)  
    fig.tight_layout()
    fig.savefig(path_out/('c-index_sig-barplot.png'), dpi=300)





    # Print selected features
    # for sig, feat in survival_sig["selected_features_names"].items():
    #     print("Selected Features", sig)
    #     for feat_split in feat:
    #         print(feat_split)





    # ---------------------- Kaplan Meier -----------------------------
    for anno  in annos:
        km_sig = survival_sig[anno]["kaplan-meier_estimate"]
        fig, axes = plt.subplots(ncols=len(km_sig), nrows=1, figsize=(15,6) ) 
        for v, sig in enumerate(km_sig.keys()):
            axis = axes[v]
            
            km_estimates = km_sig[sig]
            alpha = 1 if anno == "GT" else 0.1 
            for n, ((x_high, y_high), (x_low,y_low)) in enumerate(km_estimates):
                axis.step(x_high,y_high, where="post", c='r', alpha=alpha, linewidth=2, label="High Risk" if n==0 else None)
                axis.step(x_low,y_low, where="post", c='g', alpha=alpha, linewidth=2, label="Low Risk" if n==0 else None)
            
            median_log_rank  = np.median(survival_sig[anno]["kaplan-meier_p-value"][sig])
            print(sig, "Median KM p-value", median_log_rank )
            

            axis.set_ylabel('Survival Probability', fontdict=fontdict)
            axis.set_xlabel('Survival Time', fontdict=fontdict)
            axis.set_title("Kaplan-Meier Log-Rank Test p={:.3f}  \"{}\"".format(median_log_rank, sig), fontdict=fontdict)   
            axis.grid(False)
            axis.legend(loc='lower left', shadow=True, edgecolor='k')
            axis.spines['right'].set_visible(False)
            axis.spines['top'].set_visible(False)  

        fig.tight_layout()
        fig.savefig(path_out/('kaplan-meier_sig_'+anno.lower()+'.png'), dpi=300)



