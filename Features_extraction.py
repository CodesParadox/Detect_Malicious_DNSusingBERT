import cudf
import math
import nltk

nltk.download('words')
from nltk.corpus import words
import pandas as pd
import re
from transformers import BertTokenizer, BertModel
import time
import torch
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle

import numpy as np
import random
def count_parts(row):
    return len(row)


import cudf
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import random

# dictinary to get the corrispondig dns type to its meaning 
types_dict = {
    1: "A",
    2: "NS",
    5: "CNAME",
    28: "AAAA",
    65: "HTTPS"
}
# get the type, it it exist in types_dict 
def type_i_to_s(t):
  if t in [1,2,5,28]:
    return types_dict[t]
  return str(t)

# helper function to cacula
def simplify_answer(ans):
  simple = ""
  if ans.size > 0:
    l = len(ans)
    for i, line in enumerate(ans):
      simple += f"{line['name'][:-1]} {line['ttl']} {type_i_to_s(line['type_'])} {line['value_str']}"
      if i < l-1:
        simple += " "
  return simple


# making the formmated answer column - that we use for the classificatin model 
def get_filter_out_calculate(df):
    # filter out
    # dns name that Akamai ueses for analyzing an checking urls - hence we dont consider them as a valubale data      
    dnsq = df.loc[df.qname != 'deathstar.iamaka.net.']
    # we fillter out every url that come from the cache       
    dnsq = dnsq.loc[dnsq.l1_cache_hit == False]
    dnsq = dnsq[dnsq['answer'].apply(lambda arr: len(arr) > 0)]

    # calculate
    dnsq['duration'] = (dnsq['response_ts'] - dnsq['request_ts'])
    dnsq['cnt_ans_parts'] = dnsq['answer'].apply(lambda arr: len(arr))
    dnsq['mean_ttl'] = dnsq['answer'].apply(lambda arr: np.mean([d.get('ttl') for d in arr]))
    dnsq['cnt_ans_A'] = dnsq['answer'].apply(lambda arr: sum(d.get('type_') == 1 for d in arr))
    dnsq['cnt_ans_NS'] = dnsq['answer'].apply(lambda arr: sum(d.get('type_') == 2 for d in arr))
    dnsq['cnt_ans_CNAME'] = dnsq['answer'].apply(lambda arr: sum(d.get('type_') == 5 for d in arr))
    dnsq['cnt_ans_AAAA'] = dnsq['answer'].apply(lambda arr: sum(d.get('type_') == 28 for d in arr))
    dnsq['formatted_ans'] = dnsq["answer"].apply(simplify_answer)
    dnsq.drop_duplicates(subset=['formatted_ans'], inplace=True)

    return dnsq





# how to use
# add the list as a new column to the DataFrame
# df['entropy'] = get_entropy_column(df)
# Entropy is a concept from information theory that measures
# the amount of uncertainty or randomness in a set of data.
# In the context of entropy, data refers to a collection
# of events or observations with associated probabilities.
# Shannon's entropy formula: H(X) = - ∑ P(x) * log2(P(x))
def get_entropy_column(df):
    # calculate entropy for each string in the 'qname' column
    entropy_list = []
    for i, row in df.to_pandas().iterrows():
        qname = row['qname']
        # count the frequency of each character in the string
        char_freq = {char: qname.count(char) for char in set(qname)}

        # calculate entropy using the formula
        entropy = sum(-freq / len(qname) * math.log(freq / len(qname), 2) for freq in char_freq.values())

        # append the entropy to the list
        entropy_list.append(entropy)
    return entropy_list


# how to use
# add the list as a new column to the DataFrame
# df['type_ratio'] = df['qname'].map(get_type_ratio_column(df))
# A (Address) record is the most fundamental type of DNS record,
# it indicates the IPv4 address of a given domain name
# (IPv6 address for the case of AAAA Record).
# here we get a how many times we got 1(A) or 28(AAAA) as a qtype
def get_type_ratio_column(df):
    #  Create an empty dictionary to hold the results
    results = {}

    # Iterate over each unique value in the qname column
    for name in df['qname'].unique():
        # Get a subset of the DataFrame for the current name
        subset = df[df['qname'] == name]

        # Count how many times 1 or 28 appear in the qtype column for the current name
        count = subset['qtype'].isin([1, 28]).sum()
        ratio = count / subset['qtype'].count()
        # Add the count to the results dictionary
        results[name] = ratio
    return results


# how to use
# add the list as a new column to the DataFrame
# df['qname_vol'] = df['qname'].map(get_unique_query_volume(df))
# In a normal setting, DNS traffic is rather sparse as responses
# are largely cached within the stub resolver. However, in the
# case of data exchange over the DNS, the domain-specific traffic
# is expected to avoid cache by non-repeating messages,
# or short time-to-live in order for the data to make it to the
# attacker’s server. Avoiding cache as well a lengthy data exchange,
# might result in a higher volume of requests compared
# to a normal setting. This feature is computed as follows:
# Vol(W) = |{Q|Q ∈ W}|. the explantion from the "Detection of malicious and low throughput
# data exfiltration over the DNS protocol"
def get_unique_query_volume(df):
    #  Create an empty dictionary to hold the results
    results_vol = {}
    # Iterate over each unique value in the qname column
    for name in df['qname'].unique():
        # Get a subset of the DataFrame for the current name
        subset = df[df['qname'] == name]
        # Add the count to the results dictionary
        results_vol[name] = subset['qname'].count()
    return results_vol


# how to use
# add the list as a new column to the DataFrame
# pdf['qname_len'] = pdf['qname'].map(get_query_length(df))
# this function calculate the length of the dns query
def get_query_length(df):
    #  Create an empty dictionary to hold the results
    results_length = {}
    # Iterate over each unique value in the qname column
    for name in df['qname'].unique():
        # Add the count to the results dictionary
        results_length[name] = len(name)
    return results_length


# how to use
# add the list as a new column to the DataFrame
# df['longest_word'] = pdf['qname'].map(get_longest_word_and_length(df)[0])
# df['longest_word_len'] = df['qname'].map(get_longest_word_and_length(df)[1])
# this function calculates the English word within each unique value in the 'qname' column
# of the provided DataFrame (df). It returns two dictionary that contains the longest
# English word for each unique value and ratio between the length of the dns qname to the longest English word within.
def get_longest_word_and_length(df):
    # get english dictionary
    english_dict = set(nltk.corpus.words.words())
    #  Create an empty dictionary to hold the results
    results_longest_word = {}
    results_longest_word_len = {}

    # Iterate over each unique value in the qname column
    for name in df['qname'].unique():
        # Get a subset of the DataFrame for the current name
        subset = df[df['qname'] == name]

        subdomains = re.findall(r'\b[A-Za-z]+\b', name)
        subdomains.sort(reverse=True)

        english_word = ''

        for subdomain in subdomains:
            # Generate all possible substrings in decreasing order of length
            substrings = [subdomain[i:j] for i in range(len(subdomain)) for j in range(i + 1, len(subdomain) + 1)]
            substrings.sort(key=len, reverse=True)

            # Check if each substring is a valid English word
            for substring in substrings:
                if len(english_word) > len(substring):
                    break
                if substring.lower() in english_dict:
                    # If a substring is a valid English word, return its length
                    english_word = substring
                    break

        if len(english_word) > 0:
            results_longest_word[name] = english_word
            results_longest_word_len[name] = len(english_word) / len(name)
        else:
            results_longest_word[name] = ""
            results_longest_word_len[name] = 0
    return results_longest_word, results_longest_word_len


# how to use
# add the list as a new column to the DataFrame
# df['unique_query_ratio'] = df['qname'].map(get_unique_query_ratio(df))
# this function calculates the unique query ratio
# when you're using subdomains within a domain as messages,
# it is unlikely for them to be repeated.
# As a result, when comparing domains used for exfiltration to normal domains,
# we anticipate a significantly higher unique query ratio in the case of normal domains.
# Uniq(W)= |{Q|Q ∈ W}| / sum(overall qname)
def get_unique_query_ratio(df):
    # Create an empty dictionary to hold the results
    results_unique_query_ratio = {}

    # Iterate over each unique value in the qname column
    for name in df['qname'].unique():
        # Get a subset of the DataFrame for the current name
        subset = df[df['qname'] == name]
        count = subset['qname'].count()
        unique = subset["qname"].nunique()
        results_unique_query_ratio[name] = unique / count
    return results_unique_query_ratio


# how to use
# add the list as a new column to the DataFrame
# df['embedding_bert'] = df['qname'].map(get_bert_embedding())
# this function calculates the bert_embedding
def get_bert_embedding(df):
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    embedding_bert = {}
    # get the unique values of the qname column
    unique_qnames = df['qname'].unique()

    # iterate over the unique qnames and extract the rows that match each qname
    for qname in unique_qnames:
        embedding_bert[qname] = calculate_sentence_embedding(qname, tokenizer, model)

    return embedding_bert


# function to calculate sentence embeddings using BERT
def calculate_sentence_embedding(sentence, tokenizer, model):
    # Tokenize sentence
    tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=True)
    # Pad tokenized sequence
    max_len = 32  # set max length of the token sequence
    padded_sentence = tokenized_sentence[:max_len] + [0] * (max_len - len(tokenized_sentence))
    # Create attention mask
    attention_mask = [1 if token != 0 else 0 for token in padded_sentence]
    # Convert input to PyTorch tensors
    input_ids = torch.tensor([padded_sentence])
    attention_mask = torch.tensor([attention_mask])

    # Pass input through BERT model
    with torch.no_grad():
        embeddings = model(input_ids, attention_mask=attention_mask)
    # Get the embeddings for the [CLS] token, which represents the entire sequence
    sentence_embedding = embeddings[0][:, 0, :].numpy()
    return sentence_embedding


# ----important----
# you must pass a df that contain the column of embedding_bert
# how to use
# add the list as a new column to the DataFrame
# df['manhattan_distance_emmbading'] = calculate_distance_from_beginning(df)
# Manhattan distance, also known as L1 distance or taxicab distance,
# is a measure of the distance between two points in a coordinate system.
# The Manhattan distance between two points (x1, y1) and (x2, y2) in a two-dimensional space
# is calculated by summing the absolute differences between their corresponding coordinates:
# Manhattan distance = |x2 - x1| + |y2 - y1|
# it can also be generalized to a multi damnation like we do in our function
# the reference point that we chose is the beginning of the axles meaning [0,0,0...0,0]
def get_manhattan_distance(df):
    distances = []
    for index, row in df.iterrows():
        dd_array = row['embedding_bert']
        distance = np.sum(np.abs(dd_array))
        distances.append(distance)

    return distances


# ----important----

# you must pass a df that contain the column of embedding_bert
# how to use
# add the list as a new column to the DataFrame
# df['euclidean_distance_emmbading'] = get_manhattan_distance(df)
# The Euclidean distance between two points (x1, y1) and (x2, y2) in a two-dimensional space
# is calculated using the Pythagorean theorem, which states that the square of the hypotenuse
# of a right triangle is equal to the sum of the squares of the other two sides.
# The Euclidean distance is the square root of the sum of the squared differences
# between the coordinates:
# Euclidean distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
# it can also be generalized to a multi damnation like we do in our function
# the reference point that we chose is the beginning of the axles meaning [0,0,0...0,0]
def get_euclidean_distance(df):
    distances = []
    for index, row in df.iterrows():
        dd_array = row['embedding_bert']
        distance = np.sqrt(np.sum(np.square(dd_array)))
        distances.append(distance)
    return distances



