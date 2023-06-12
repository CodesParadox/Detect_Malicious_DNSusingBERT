import os
import pandas as pd


directory = 'D:\\DeepLearning DNS Akamai\\ariel_university_axLogs'
file_list = []

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    file_list.append(f)

df = pd.read_parquet(file_list[0])
df['File'] = file_list[0].split('\\')[-1]
for i in range(0, 2):
    print(i)
    curr_df = pd.read_parquet(file_list[i])
    curr_df['File'] = file_list[i].split('\\')[-1]
    df = pd.concat([df, curr_df])

csv_out = r'D:\\DeepLearning DNS Acamai\\DatasetDNS.csv'
df.to_csv(csv_out, index=False)

# # reset the index
# df.reset_index(drop=True, inplace=True)
#
#
# # convert the dataframe to a json file
# json_data = df.to_json()
#
# # save the json file to the code directory
# with open('dns_parquet_files.json', 'w') as f:
#     f.write(json_data)


#pd.to_pickle(df, "dns_parquet_files2.pkl")
