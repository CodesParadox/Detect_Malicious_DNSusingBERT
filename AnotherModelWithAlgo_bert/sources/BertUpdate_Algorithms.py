import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import Levenshtein
from gensim.models import Word2Vec
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertForSequenceClassification

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# Load the DNS traffic dataset and preprocess the data
df = pd.read_pickle('../dns_final_files.pkl')

# Extract relevant features from the dataset and convert them to tensors
request_ts = torch.tensor(df['request_ts'].values).to(device)
response_ts = torch.tensor(df['response_ts'].values).to(device)
client_token_dec = torch.tensor(df['client_token_dec'].values).to(device)
qname = torch.tensor(df['qname'].values)
qtype = torch.tensor(df['qtype'].values)
answer = torch.tensor(df['answer'].values).to(device)
nx_domains = torch.tensor(df['nx_domains'].values).to(device)
l1_cache_hit = torch.tensor(df['l1_cache_hit'].values).to(device)
response_size = torch.tensor(df['response_size'].values).to(device)
category_id = torch.tensor(df['category_id'].values).to(device)
in_BL = torch.tensor(df['in_BL'].values).to(device)

# Calculate additional features using algorithms
levenshtein_distances = [Levenshtein.distance(qname_val, qtype_val) for qname_val, qtype_val in zip(qname, qtype)]
levenshtein_distances_normalized = [(val - np.min(levenshtein_distances)) / (np.max(levenshtein_distances) - np.min(levenshtein_distances)) for val in levenshtein_distances]
levenshtein_distances_normalized = torch.tensor(levenshtein_distances_normalized).to(device)

qname_tokens = [qname_val.split() for qname_val in qname]
word2vec_model = Word2Vec(qname_tokens, size=100, window=5, min_count=1, workers=4).to(device)

# Encode the features into input tensors for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased').to(device)
encoded_inputs = tokenizer(request_ts.tolist(), response_ts.tolist(), client_token_dec.tolist(), qname.tolist(), qtype.tolist(), answer.tolist(), nx_domains.tolist(),
                           l1_cache_hit.tolist(), response_size.tolist(), category_id.tolist(), levenshtein_distances_normalized.tolist(),
                           padding=True, truncation=True, return_tensors='pt').to(device)

# Create a TensorDataset
dataset = TensorDataset(encoded_inputs['input_ids'].to(device), encoded_inputs['token_type_ids'].to(device), encoded_inputs['attention_mask'].to(device),
                        torch.tensor(in_BL).to(device))

# Split the dataset into training and testing sets
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Create a dataloader for the training set
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create a dataloader for the testing set
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Load the pre-trained BERT model
pretrained_model_name = 'bert-base-uncased'
pretrained_model = BertForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=2).to(device)

# Set the optimizer
optimizer = torch.optim.AdamW(pretrained_model.parameters(), lr=2e-5)

# Set the loss function
criterion = nn.CrossEntropyLoss()

# Train the model
pretrained_model.train()
for epoch in range(10):
    for batch in train_dataloader:
        # Get the inputs
        input_ids = batch[0].to(device)
        token_type_ids = batch[1].to(device)
        attention_mask = batch[2].to(device)
        labels = batch[3].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = pretrained_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)

        # Compute the loss
        loss = criterion(outputs[1], labels)

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

    print(f'Epoch {epoch+1} loss: {loss.item()}')

# Test the model
pretrained_model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch in test_dataloader:
        # Get the inputs
        input_ids = batch[0].to(device)
        token_type_ids = batch[1].to(device)
        attention_mask = batch[2].to(device)
        labels = batch[3].to(device)

        # Forward pass
        outputs = pretrained_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # Get the predictions
        _, predicted = torch.max(outputs[0], 1)

        # Update the counters
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy: {correct / total}')

# Save the model
torch.save(pretrained_model.state_dict(), 'model.pt')
