import pandas as pd
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


#read the DataFrame from the pkl file
df = pd.read_parquet('filter/dnsq_ben_all.parquet')
#df = pd.

#view the DataFrame as gui
df.head()
#read the csv file to the DataFrame
#df = pd.read_csv("D:\DeepLearning DNS Akamai\separate\dns_parquet_files.csv")


#df = pd.read_pickle("dns_parquet_files.pkl")

# df = pd.read_pickle('dns_final_files.pkl')
# profile = ProfileReport(df, title='DNS Profiling Report')
# profile.to_file('dns_report.html')

