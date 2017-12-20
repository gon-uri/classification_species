import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tsfresh as ts
import pandas as pd
from tsfresh import extract_relevant_features
#from tsfresh import extract_features
#from tsfresh import select_features
#from tsfresh.utilities.dataframe_functions import impute

timeseries = pd.read_csv('/home/usuario/Desktop/data_grupo1/amp_ent_2.txt', sep='\t')
y = pd.read_csv('/home/usuario/Desktop/data_grupo1/amp_ent_2_labels.txt', sep='\t', header=None)
#y_array = y.as_matrix()
features_filtered_direct = extract_relevant_features(timeseries, y[1], column_id='id', column_sort='time')

#extracted_features = extract_features(timeseries, column_id="id", column_sort="time")
#impute(extracted_features)
#features_filtered = select_features(extracted_features, y)