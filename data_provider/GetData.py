import os.path
import pandas as pd
import numpy as np
import random


def makeMissingData(path, missing_ratio, seed, missingness='MCAR'):
    df = pd.read_csv(path)
    return makeMissingDataByDf(df=df, missing_ratio=missing_ratio, seed=seed, missingness=missingness)



def makeMissingDataByDf(df, missing_ratio, seed, missingness='MCAR'):
    np.random.seed(seed)

    print("seed and rate:", seed, missing_ratio)
    
    mask = np.random.rand(*df.shape)
    df2 = df.copy()
    
    mask[mask <= missing_ratio] = 0     # masked
    mask[mask != 0] = 1                 # remained
    
    df2[mask == 0] = np.nan
    print(df.shape[0]*df.shape[1], (1-mask).sum())

    return df, df2





