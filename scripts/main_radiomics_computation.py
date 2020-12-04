from pathlib import Path
import radiomics
from radiomics import featureextractor
import csv
import json
import SimpleITK as sitk
import numpy as np 
import h5py
import logging
import sys 
import reliableradiomics.metrics.statistics as stats

logger = logging.getLogger(__name__)



def get_major_idxs(lesion_idxs, lesions, des_num_slices=-1):
    """Returns the desired number of lesion indices with decent number of nonzero label slices"""
    ava_num_slices = len(lesion_idxs)

    if des_num_slices == -1: 
        des_num_slices = ava_num_slices
    
    if ava_num_slices < des_num_slices:
        logger.warning("Only {} of {} slices are available".format(ava_num_slices, des_num_slices))
        des_num_slices = ava_num_slices
    
    non_zeros = np.count_nonzero(lesions, axis=tuple(range(1,lesions.ndim)))
    lesion_idxs = lesion_idxs[np.argsort(-non_zeros)] # descending order 
    return [lesion_idxs[n] for n in range(des_num_slices)]


def extract_features(uid, image, label_anots, pred_anots, spacing, extractor, min_pred_anno, path_out, errors, dice_min=0):
    """Try to extract features and write them into a file, otherwise return specific error code"""
    # Create vector with featurenames 
    featureVectors = [] # ["id"]+featurenames
    spacing = spacing.astype(np.float) 

    # Create input image with spacing 
    image_img = sitk.GetImageFromArray(image, isVector=False) # z,y,x -> x,y,z 
    image_img.SetSpacing(spacing) # x,y,z 

    # Check if expert-annotators have minimum dice-score agreement among themselves  
    dice_scores = [] 
    for anno_n, gt_mask_m in enumerate(label_anots[:-1]): 
        for gt_mask_n in label_anots[anno_n+1:]:
            dice_scores.append(stats.dice(gt_mask_m.flatten(), gt_mask_n.flatten(), average='binary'))
    
    if np.any(np.asarray(dice_scores)<dice_min ):
        logger.warning("Must exclude {} because expert-annotators have not minimum dice-score agreement {} among themselves: {}".format(uid, dice_min, dice_scores))
        return 1

    # Try to extract label features 
    for mask_num, mask_np in enumerate(label_anots):
        # Recover spacing 
        mask = sitk.GetImageFromArray(mask_np, isVector=False)
        mask.SetSpacing(spacing) 
        try:
            featureVector = extractor.execute(image_img, mask)
            # Check if feature vector contains any NaN-values
            if np.isnan([val for val in featureVector.values() if isinstance(val,np.ndarray)]).any():
                logger.info("Must exclude label {} annotation {} due to {}.".format(uid, mask_num, "NaN value(s)"))
                logger.warning("Must exclude {} because there were not enough valid features extractable from expert annotations".format(uid))
                return 1
            featureVectors.append([uid+"_"+str(mask_num)]+list(featureVector.values()))
        except BaseException as error:
            logger.info("Must exclude label {} annotation {} due to {}.".format(uid, mask_num, error))
            logger.warning("Must exclude {} because there were not enough valid features extractable from expert annotations".format(uid))
            return 1 # Stop this entire slice: All label annotations must be valid but at least one already failed 

  
    # Try to extract Prediction features for this specific lesion and slide 
    for mask_num, mask_np in enumerate(pred_anots):

        # Check if prediction-annotator has minimum dice agreement with gt-annotator(s)
        dice_scores = [] 
        for gt_mask in label_anots:
            dice_scores.append(stats.dice(gt_mask.flatten(), mask_np.flatten(), average='binary'))
        if np.all(np.asarray(dice_scores)<dice_min):
            error = "Prediction-annotator can not reach the minimum dice score {} given by the gt annoators: {}".format(dice_min, dice_scores)
            logger.info("Must exclude prediction {} annotation {} due to {}.".format(uid, mask_num, error))
            continue

        # Recover spacing 
        mask = sitk.GetImageFromArray(mask_np, isVector=False)
        mask.SetSpacing(spacing) 
        try:
            featureVector = extractor.execute(image_img, mask)
            # Check if feature vector contains any NaN-values
            if np.isnan([val for val in featureVector.values() if isinstance(val,np.ndarray)]).any():
                logger.info("Must exclude prediction {} annotation {} due to {}.".format(uid, mask_num, "NaN value(s)"))
                continue
            featureVectors.append([uid+"_"+str(mask_num)]+list(featureVector.values()))
        except BaseException as error:
            logger.info("Must exclude prediction {} annotation {} due to {}.".format(uid, mask_num, error))
            continue # Just skip this annotation, other prediction annotations might be valid 
        
        # Skip process if desire number is reached 
        if len(featureVectors)>= label_anots.shape[0]+min_pred_anno:
            featureVectors.insert(0, ["id"]+list(featureVector.keys()))
            # Append feature vector to file 
            with open(path_out, 'a') as f:
                results = zip(*featureVectors)
                result_writer = csv.writer(f, lineterminator='\n')
                result_writer.writerows(results)  
            return 0 # no error 
        

    logger.warning("Must exclude {} because there were not enough valid features extractable from prediction annotations".format(uid))
    return 2 # error predcition annotations  
  


def main_feat_ext(paramsFile, path_data, path_out):
    #  ----- General -------     
    path_feat_file = path_out/'radiomics.csv'
    path_stats = path_out/'radiomics_stats.json'

    path_out.mkdir(parents=True, exist_ok=True)

    dice_min = 0.3 # Minimum dice score that gt-raters must reach within all gt-raters and pred-rater compared to at least one gt-rater
    min_pred_anno = 100 # Minimum number of valid feature extractions from prediction annotations per subject, else reject  
    des_num_slices =  5 # Max number of other slices that might be used after no features could be extracted (-1 for all)

    # Logging
    class ModuleFilter(object):
        def filter(self, record):
            modules = ['__main__',] #  'radiomics'
            if any([record.name.find(mod, 0, len(mod)) == 0 for mod in modules]):
                return True
            return False 
    
    s_handler = logging.StreamHandler(sys.stdout)
    s_handler.addFilter(ModuleFilter())
    f_handler = logging.FileHandler(path_out/'radiomics_comp_log.log', 'w')
    f_handler.addFilter(ModuleFilter())
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[s_handler, f_handler])
    

    # Initialize feature extractor
    assert paramsFile.is_file(), "Error parameter file not found"
    extractor = featureextractor.RadiomicsFeatureExtractor(str(paramsFile))
    

    # -------- Execute feature extraction -------- 
    logger.info("Start calculating features")
    with h5py.File(path_data, 'r') as f:
        
        # Read File   
        uids = f['uids']              # [Samples, 1]
        uids = np.asarray([uid.tobytes().decode('ascii') for uid in uids]) 
        spacings = f['spacing']         # [Samples, 3] with (x,y,z)
        images = f['images']            # [Samples, z,y,x ]
        labels = f['labels']            # [Samples, Annotations, z,y,x]
        predictions = f['predictions']  # [Samples, Annotations, z,y,x, Probabilities ]

        
        # Create output file and write headline
        num_label_anno = labels.shape[1]
        headline = ["Features"]+ \
                   ["GT_"+str(n+1) for n in range(num_label_anno)]+ \
                   ["Pred_"+str(n+1) for n in range(min_pred_anno)]
        with open(path_feat_file, 'w') as f:
                result_writer = csv.writer(f, lineterminator='\n')
                result_writer.writerow(headline) 
        
        # Log statics
        statistics = {} 
        statistics['num_pred_anno'] =  min_pred_anno
        statistics['num_expert_anno'] = num_label_anno
        statistics['num_subjects'] = len(uids)
        statistics['num_patients'] = len(np.unique([uid.split('_',1)[0] for uid in uids]))

        # ------ Iterate over lesions slides (Option 1)
        # Split uids into [lesion_id, slice_idx]
        ids =  np.asarray([uid.rsplit('_',1) for uid in uids])
        # Create a list of arrays, each array contains indices belonging to the same lesion 
        lesion_uids = np.unique(ids[:,0])
        lesions_idxs =  [np.where(ids[:,0]==i)[0] for i in lesion_uids] 

        # Statistics 
        statistics['num_lesions'] = len(lesion_uids)
        statistics['num_slices_per_lesion'] = [len(slices) for slices in lesions_idxs] 
        statistics['error_attempts'] = []
        
        for lesion_idxs in lesions_idxs:
            # Decent order of nonzero slices   
            lesion_center_idxs = get_major_idxs(lesion_idxs, labels[lesion_idxs], des_num_slices)

            # Try to extract features from around the center 
            attempts = 0
            error = True 
            for lesion_center_idx in lesion_center_idxs:
                uid = str(uids[lesion_center_idx])
                spacing = spacings[lesion_center_idx]
                image = images[lesion_center_idx]
                label_anots = labels[lesion_center_idx]
                pred_anots = np.argmax(predictions[lesion_center_idx],  axis=-1)

                # (Workaround for PyRadiomics)  Range [-0.5, 0.5] => [-500, 500]
                # image = image*1000
                
                # Try to extract valid features and if possible save to file otherwise return error code > 0 
                error = extract_features(uid, image, label_anots, pred_anots, spacing, extractor, min_pred_anno, path_feat_file, statistics, dice_min)
                if not error:
                    break
                else:
                    attempts += 1 
                  
            if error:
                lesion_id = ids[lesion_idxs][0,0]
                logger.warning("Must exclude entire lesion {}".format(lesion_id))
            
            # Statistics 
            statistics['error_attempts'].append([error, attempts])
                
                
        # Write statistics to file      
        with open(path_stats, 'w') as f:
            json.dump(statistics, f)

        logger.info("Finish calculating features")       

              


if __name__ == "__main__":
    dataset = 'nsclc'
    paramsFile = Path.cwd()/('data/settings/params_'+dataset+'.yaml')
    path_data = Path.cwd()/('data/images/'+dataset+'/images.hdf5')
    path_out = Path.cwd()/('results/features/'+dataset)

    
    main_feat_ext(paramsFile, path_data, path_out)