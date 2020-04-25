# SHP6004_Project_2
Model building for MIMIC Dataset

# Data preprocessing files
  * For baseline 24-hour point data
  
        1. BaselineData_preprocess.R 
  * For Time series data
  
        1. Data_preprocess_DiasBP.R
        2. Data_preprocess_HeartRate.R
        3. Data_preprocess_MeanBP.R
        4. Data_preprocess_SpO2.R
        4. Data_preprocess_SysBP.R
        5. Data_preprocess_TempC.R
        6. Combine_TimeSeriesData.R
        7. Combine_TimeSeriesData_NOzeros.R

# Models using Time series data
  * Overall in-ICU mortality prediction
  
        Mortality_Classification_Time_Series_Data.ipynb
        
  * LOS > 48 hours classification for Patients who died in the ICU
        
        Death_LOS_Classification(48_hours)_Time_Series_Data.ipynb
  
  * LOS > 48 hours classification for Patients who were discharged from the ICU
  
        Discharge_LOS_Classification(48_hours)_Time_Series_Data.ipynb
# Models using Baseline data 
 * LOS predisction (Classification), predicting LOS within 30 days in time intervals (1-3, 4-7 or 8-30 days)
        
        LOS_Classification.py
 * LOS prediction (Regression), prediction of LOS within 30 days
        
        LOS_Regression.py
