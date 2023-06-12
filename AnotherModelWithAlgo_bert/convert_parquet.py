import pandas as pd
import os
import pickle


# set the directory path where the pkl files are stored
directory_path = 'C:/Users/Gigabyte/PycharmProjects/DNS Malwares Detection'

# create an empty list to store the dataframes
dataframes = []

# loop through the files in the directory and read them using the pd.read_pickle function
for filename in os.listdir(directory_path):
    if filename.startswith('dns_parquet_files') and filename.endswith('.pkl'):
        file_path = os.path.join(directory_path, filename)
        try:
            with open(file_path, 'rb') as f:
                df = pickle.load(f)
                dataframes.append(df)
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Error reading {filename}: {e}")

# concatenate the dataframes into a single dataframe
combined_df = pd.concat(dataframes)

# save the combined dataframe as a pkl file
output_file_path = os.path.join(directory_path, 'dns_final_files.pkl')
combined_df.to_pickle(output_file_path)

# import glob
# import pandas as pd

# # Get all the pkl files in the current directory
# all_files = glob.glob("*.pkl")

# # Combine all the pkl files into a single dataframe
# combined_df = pd.concat([pd.read_pickle(f) for f in all_files])

# # Save the combined dataframe as a new pkl file
# combined_df.to_pickle("combined.pkl")