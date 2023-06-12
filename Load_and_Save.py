import numpy as np
import pandas as pd


def load_df(fileName="df_include_added_features.pkl"):
    return pd.read_pickle(fileName)


def load_from_pqrquet(fileName="parquet_files/dns_parquet_files_20.pkl"):
    return pd.read_pickle(fileName)


# ----important----
# you must pass a df that contain the column of embedding_bert
# this function takes the 'embedding_bert' column of the DataFrame df,
# which contains multidimensional arrays as elements,
# flattens those arrays into 1D arrays,
# and converts them into individual columns of a new DataFrame.
# we use this function in order to send it to our deep learning model
def df_to_embading_bert_as_df(df):
    return pd.DataFrame(df['embedding_bert'].apply(np.ravel).apply(pd.Series))

# save the pandas dataframe as pickle
def save_df(df, file_name):
    df.to_pickle(file_name)
