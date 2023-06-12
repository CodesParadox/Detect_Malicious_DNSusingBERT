import cudf
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import random

types_dict = {
    1: "A",
    2: "NS",
    5: "CNAME",
    28: "AAAA",
    65: "HTTPS"
}

def type_i_to_s(t):
  if t in [1,2,5,28]:
    return types_dict[t]
  return str(t)

def simplify_answer(ans):
  simple = ""
  if ans.size > 0:
    l = len(ans)
    for i, line in enumerate(ans):
      simple += f"{line['name'][:-1]} {line['ttl']} {type_i_to_s(line['type_'])} {line['value_str']}"
      if i < l-1:
        simple += " "
  return simple


#פונקצית עזר למציאת הFORMATED ANSWER שאנחנו עובדים עליו
import random

def get_filter_out_calculate(df):
    # filter out
    dnsq = df.loc[df.qname != 'deathstar.iamaka.net.']
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

#המשך של בניית התשובה המפורמטת
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset

model_name = 'bert-base-uncased'  # Example model name
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

max_length = 15

# Assuming you have your dataset as a list of texts and corresponding labels
# texts = list_answer  # List of texts
texts = df_all["formatted_ans"].to_list()  # List of texts

labels = [int(label) for label in df_all["is_malicious"].to_list()]  # List of labels
texts, labels = shuffle_lists(texts, labels)

# Tokenize the texts and convert them to input tensors
input_ids = []
attention_masks = []

for text in texts:
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids.append(encoded_text['input_ids'])
    attention_masks.append(encoded_text['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Define the split ratio
train_ratio = 0.8
train_size = int(train_ratio * len(texts))

# Split the dataset
train_inputs, val_inputs = input_ids[:train_size], input_ids[train_size:]
train_masks, val_masks = attention_masks[:train_size], attention_masks[train_size:]
train_labels, val_labels = labels[:train_size], labels[train_size:]

# Create PyTorch DataLoader for training and validation sets
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
val_dataset = TensorDataset(val_inputs, val_masks, val_labels)

batch_size = 32  # changed from 16 -> 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# Move the model to the device
model.to(device)

# Define the optimizer and learning rate
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
num_epochs = 6

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        input_ids = batch[0].to(device)
        attention_masks = batch[1].to(device)
        labels = batch[2].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{num_epochs} - Average Training Loss: {avg_loss:.4f}')

# Save the fine-tuned model
model.save_pretrained('/media/hodefi/q/DNS_DL/code/model/bert_model2.sav')

# Load the fine-tuned model
model = BertForSequenceClassification.from_pretrained('/media/hodefi/q/DNS_DL/code/model/bert_model2.sav')

import pickle
from sklearn.ensemble import IsolationForest
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import sys

# Move the model to the device
model.to(device)

# Evaluation loop
model.eval()
predictions = []
true_labels = []

for batch in val_dataloader:
    input_ids = batch[0].to(device)
    attention_masks = batch[1].to(device)
    labels = batch[2]
    true_labels.extend(labels.numpy())

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_labels = torch.argmax(probabilities, dim=1)
    predictions.extend(predicted_labels.cpu().numpy())

# Perform analysis on the predictions and true labels

# Convert predictions and true labels to numpy arrays
predictions = np.array(predictions)
true_labels = np.array(true_labels)

# Generate classification report
report = classification_report(true_labels, predictions)
print(report)

# Calculate the confusion matrix
cm = confusion_matrix(true_labels, predictions)

# print the confusion matrix
print(cm)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Create a heatmap
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)

# Set labels, title, and ticks
ax.set_xlabel("Predicted Labels")
ax.set_ylabel("True Labels")
ax.set_title("Confusion Matrix")
# ax.xaxis.set_ticklabels(["Class 0", "Class 1", "Class 2"])
# ax.yaxis.set_ticklabels(["Class 0", "Class 1", "Class 2"])

# Display the plot
plt.show

#וזה בניית המודל של BERT על התשובה המפורמטת כולל הערכה

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
#בקוד זה עובד על הCPU אבל אם הGPU והCUDA עובדים לך אפשר להריץ שם

#מריצים רק על היוניק פורמטד אנסר
