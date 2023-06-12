import pandas as pd
import pyarrow.parquet as pq
import os

# set the directory path where the parquet files are stored
directory_path = 'D:\\DeepLearning DNS Acamai\\ariel_university_axLogs'

# create an empty list to store the dataframes
dataframes = []

#counter for the files
i=0

# loop through the files in the directory and read them using the pq.read_table function
for filename in os.listdir(directory_path):
    print(i)
    if filename.endswith('.parquet'):
        file_path = os.path.join(directory_path, filename)
        table = pq.read_table(file_path)
        df = table.to_pandas()
        dataframes.append(df)
        i+=1


# combine the dataframes into a single dataframe
combined_df = pd.concat(dataframes)

# reset index of combined dataframe and drop old index
combined_df.reset_index(drop=True, inplace=True)



# convert the combined dataframe to a csv file
csv_data = combined_df.to_csv()

# save the csv file to the code directory
with open('D:\DeepLearning DNS Akamai\DatasetDNS.csv', 'w') as f:
    f.write(csv_data)



# # convert the combined dataframe to a json file
# json_data = combined_df.to_json()

# # save the json file to the code directory
# with open('DatasetDNS.json', 'w') as f:
#     f.write(json_data)