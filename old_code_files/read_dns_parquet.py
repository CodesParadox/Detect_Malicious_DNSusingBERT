import os
import pandas as pd

directory = "/home/hodefi/projects/DNS_DL/ariel_university_axLogs-20230327T122453Z-001"
file_list = []

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    file_list.append(f)

df = pd.read_parquet(file_list[0])
df['File'] = file_list[0].split('\\')[-1]

for i in range(1, len(file_list)):  # (1, 20)
    if i % 10 == 0:
        pd.to_pickle(df, "dns_parquet_files_" + str(i) + ".pkl")
        print(str(i) + "done!")
        df = pd.DataFrame(columns=df.columns)

    else:

        # print(file_list[i])
        curr_df = pd.read_parquet(file_list[i])
        curr_df['File'] = file_list[i].split('\\')[-1]
        df = pd.concat([df, curr_df])

pd.to_pickle(df, "dns_parquet_files_" + "last" + ".pkl")
print("last" + "done!")
print("finish")