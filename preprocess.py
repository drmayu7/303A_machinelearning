#import numpy as np
import pandas as pd

def clean_impute(df):
    #data_cleaning
    #name header
    header_dict = {
        0:'sample_id',
        1:'cl_thcknss',
        2:'size_cell_un',
        3:'shape_cell_un',
        4:'marg_adhesion',
        5:'size_cell_single',
        6:'bare_nucl',
        7:'bl_chrmatn',
        8:'nrml_nucleo',
        9:'mitoses',
        10:'class'
    }
    df.rename(columns=header_dict,inplace=True)
    print('Columns renaming successful')

    #replace ? values with null
    df.replace({'?':'NaN'},inplace=True)
    print("All '?' values replaced with 'NaN'")

    #save and re-read csv
    df.to_csv('Dataset/processed/breast-cancer-wisconsin.csv',index=False)
    df = pd.read_csv('Dataset/processed/breast-cancer-wisconsin.csv')

    #imputation of null values
    #from numpy import isnan
    #from pandas import read_csv
    #from sklearn.impute import SimpleImputer

    #data = df.values
    #totalCol = data.shape[1]
    #ix = [i for i in range(totalCol) if (i !=10) & (i!=0)]
    #X = data[:,ix]
    #y = data[:,10]
    #z = data[:,0]

    #print('Missing: %d' % sum(isnan(X).flatten()))

    #imputer = SimpleImputer(strategy='mean')
    #imputer.fit(X)
    #Xtrans = imputer.transform(X)

    #print('Missing: %d' % sum(isnan(Xtrans).flatten()))
    #new_y = np.atleast_2d(y).T
    #new_z = np.atleast_2d(z).T

    #df_trans = np.column_stack((Xtrans,new_y,new_z))

    print(f'''
Calculate missing values from dataframe
{df.isnull().sum()}
''')

    ##Other Imputation Method
    # Imputing with MICE
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn import linear_model

    df_mice = df.filter(['cl_thcknss','size_cell_un','shape_cell_un','marg_adhesion','size_cell_single','bare_nucl','bl_chrmatn','nrml_nucleo','mitoses'], axis=1).copy()

    # Define MICE Imputer and fill missing values
    mice_imputer = IterativeImputer(estimator=linear_model.BayesianRidge(), n_nearest_features=None, imputation_order='ascending')

    df_mice_imputed = pd.DataFrame(mice_imputer.fit_transform(df_mice), columns=df_mice.columns)

    # Remerge sample_id and class column into imputed dataframe
    extracted_col = df[['sample_id','class']]
    df_mice_imputed = df_mice_imputed.join(extracted_col)

    # Resort columns
    df_mice_imputed = df_mice_imputed[['sample_id','cl_thcknss', 'size_cell_un', 'shape_cell_un', 'marg_adhesion',
           'size_cell_single', 'bare_nucl', 'bl_chrmatn', 'nrml_nucleo', 'mitoses','class']]

    print('Null values imputated using Multivariate Imputation by Chained Equation (MICE) method')
    print(f'''
Calculate missing values from imputated dataframe
{df_mice_imputed.isnull().sum()}
''')

    return df_mice_imputed

