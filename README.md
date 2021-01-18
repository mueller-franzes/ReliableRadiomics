# Reliable Radiomics 

This is the code for the paper "Reliable Radiomics – Radiomic Reliability Analysis can Enhance Radiomic Machine Learning Algorithms" 

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

## Step 2: Data

* Put your segmented images under /data/images/$DatasetName/images.hdf
* Put your radiomics settings under  /data/settings/
* Put your survival data under /data/images/$DatasetName/survival.csv

Make sure your images.hdf has the following keys and shape:
* uids   ~ [Samples, ] with PatientID_LesionIdx_SliceIdx
* images ~ [Samples, z,y,x]
* labels ~ [Samples, Annotations, z,y,x]
* spacing ~ [Samples , 3] with order (x,y,z) 

## Step 3: Execute 

1.  main_radiomics_computation.py
2. main_radiomics_evaluation.py
3. main_radiomics_evaluation_between.py
4. main_survival_computation.py
5. main_survival_evaluation.py

* Some minor code changes (setting the correct $DatasetName) at the beginning of each .py file may be required to use your specific dataset
