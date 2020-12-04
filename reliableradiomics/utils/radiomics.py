import pandas as pd
import numpy as np 
import h5py 
import json


def load_features(path_feat_file, anno_names=['GT', 'Pred'], 
                  exclude_features=[]):
    """Load feature values from .csv and group them by annotators and feature name"""
    # Example: exclude_features=['Flatness', 'LeastAxisLength']
    df = pd.read_csv(path_feat_file) # Note: dtype=float not working here as some rows are strings 
    col_names = list(df)
    select_cols = {anno:[ n-1 for n, col_name in enumerate(col_names) if col_name.startswith(anno)] 
                    for anno in anno_names}
    dfg = df.groupby('Features')
    df_feat = pd.DataFrame(dfg)
 
    # Select all features which were computed for a specific image type 
    df_imgtype = df_feat.loc[~df_feat.iloc[:,0].str.startswith('diagnostics') | (df_feat.iloc[:, 0] == "id")  ]

    # Transfer data into array and split by annotator 
    feature_values = {anno:{} for anno in anno_names}  
    feature_uids = {}  
    for _, row in df_imgtype.iterrows():
        # Get feature name 
        name = row[0]

        # Separate UID from features 
        if name == "id":
            for anno in anno_names:
                feature_uids[anno] = row[1].iloc[:,1:].to_numpy()[:, select_cols[anno]]
            continue

        # Exclude feature if desired 
        if name.rsplit('_',1)[1] in exclude_features:
            continue 

        # Get feature values 
        values = row[1].iloc[:,1:].astype(float)
        #values = values.dropna() # WARNING: Should not be necessary  
        values = values.to_numpy()
        
        for anno in anno_names:
            feature_values[anno][name] = values[:, select_cols[anno]]

    return feature_values, feature_uids, select_cols


def load_feature_evaluation(path_eval_data):   
    rad_stats = {}
    with open(path_eval_data, 'r') as f:
        rad_stats = json.load(f)
    return rad_stats