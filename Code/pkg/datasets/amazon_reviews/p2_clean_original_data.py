import pandas as pd
from Code.pkg.data_processing.cleaning import data_cleaning

if __name__ == '__main__':
    # --------- READING DATA ----------
    data = pd.read_csv("../processed_data/OriginalData.csv", encoding="ISO-8859-1")

    # Applying data cleaning processes
    data['clean_data'] = data['text_data'].apply(data_cleaning)
    new_data = data['clean_data']
    new_data.to_csv(path_or_buf='../processed_data/CleanData.csv',
                    index=False, header=['clean_data'])
