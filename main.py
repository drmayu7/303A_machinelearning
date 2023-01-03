import pandas as pd
import preprocess

breast_ca = pd.read_csv('Dataset/breast-cancer-wisconsin.data',index_col=None,header=None)
df_impute = preprocess.clean_impute(breast_ca)