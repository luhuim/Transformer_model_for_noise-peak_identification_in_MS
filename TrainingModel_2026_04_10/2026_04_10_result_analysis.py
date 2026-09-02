#!/usr/bin/env python3
#%% Import package
import pandas as pd
pd.set_option('display.max_rows', None)
import matplotlib.pyplot as plt
import numpy as np

#%% Validation set metrics
df_2 = pd.read_csv("MasterThesis/RunOnServer/TrainingModel_2026_04_10/Epoch:13_Precision_Recall_DataFrame.tsv",sep = '\t')
# print(df_2)
#%%
print(df_2["recall"].mean())
#%%
