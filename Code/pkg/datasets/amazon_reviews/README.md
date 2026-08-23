**Empty directories in datasets**:
- When train.py trains machine learning models, it requires access to four directories which it uses to store resources
- The subdirectories in this folder are intially empty, however they are populated as models are trained
- There are also two csv files expected in this folder, OriginalData.csv and CleanData.csv - they are produced using the scripts in processing_scripts, as long as the source data exists