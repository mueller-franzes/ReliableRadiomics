# Reliable Radiomics 

This is the code for the paper "Reliability as a Precondition for Trust –Segmentation Reliability Analysis of Radiomic Features Improves Survival Prediction" 

If you use the results or code, please cite the following paper:
```
@article{
}
```

# Results 
The ICC(1) value for each dataset and feature can be found in `reliability.json` (median with 95%-confidence interval) or in `reliability.csv` (only median).  



# Run with your own data

## Step 1: Install 
    cd .../ReliableRadiomics/
    pip install numpy 
    pip install . 

## Step 2: Data (/data)

* Put your segmented images under /data/images/$DatasetName/images.hdf
* Put your radiomics settings under  /data/settings/
* Put your survival data under /data/images/$DatasetName/survival.csv

Make sure your images.hdf has the following keys and shape:
* uids   ~ [Samples, ] with PatientID_LesionIdx_SliceIdx
* images ~ [Samples, z,y,x]
* labels ~ [Samples, Annotations, z,y,x]
* spacing ~ [Samples , 3] with order (x,y,z) 

## Step 3: Execute (/scripts)
1. [main_radiomics_computation.py](scripts/main_radiomics_computation.py) 
    * Script for calculating the Radiomic features. Note that features are only calculated from segmentation that exceeds the `dice_min` threshold. 
2. [main_radiomics_evaluation.py](scripts/main_radiomics_evaluation.py) 
    * Script for calculating and visualizing the inter-rater reliability (ICC scores) from the Radiomic features. 
3. [main_radiomics_evaluation_between.py](scripts/main_radiomics_evaluation_between.py) 
    * Script to calculate and visualize the inter-rater reliability (ICC scores) of the Radiomic features across multiple datasets. 
4. [main_survival_computation.py](scripts/main_survival_computation.py) 
    * Script to estimate overall survival using a Cox model. 
5. [main_survival_evaluation.py](scripts/main_survival_evaluation.py) 
    * Script to visualize the variance of survival predictions as a function of inter-rater reliability. 
   
* Note: Some minor code changes (setting the correct $DatasetName) at the beginning of each .py file may be required to use your specific dataset
