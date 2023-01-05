import pandas as pd
import preprocess

## Read the wisconsin dataset
breast_ca = pd.read_csv('Dataset/breast-cancer-wisconsin.data',index_col=None,header=None)

## Run clean_impute function on dataset and save dataset
df_impute = preprocess.clean_impute(breast_ca)
df_impute.to_csv('Dataset/processed/breast-cancer-wisconsin-imputed.csv',index=False)
