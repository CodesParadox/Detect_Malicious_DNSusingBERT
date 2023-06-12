import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport

df = pd.read_pickle("dns_parquet_files.pkl")
profile = ProfileReport(df, title="Profiling Report",minimal=True)
# profile.to_notebook_iframe()
profile.to_file("your_report_minimal.html")