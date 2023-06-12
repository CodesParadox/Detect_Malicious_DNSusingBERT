import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DNSDataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def preprocess_data(self):
        # Preprocess the data
        # Combine relevant features into a single textual feature
        self.X = [
            f"Request_ts: {ts} Response_ts: {rts} Client_token: {client_token} Qname: {qname} Qtype: {qtype} Qclass: {qclass} Response_flags: {response_flags} Answer: {answer} Nx_domains: {nx_domains} L1_cache_hit: {l1_cache_hit} Response_size: {response_size}"
            for
            ts, rts, client_token, qname, qtype, qclass, response_flags, answer, nx_domains, l1_cache_hit, response_size, in_BL
            in zip(self.X['Request_ts'], self.X['Response_ts'], self.X['client_token_dec'], self.X['qname'],
                   self.X['qtype'], self.X['qclass'], self.X['response_flags'], self.X['answer'], self.X['Nx_domains'],
                   self.X['L1_cache_hit'], self.X['response_size'], self.X['in_BL'])]

        # Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def encode_labels(self):
        label_encoder = LabelEncoder()
        self.y_train_encoded = label_encoder.fit_transform(self.y_train)
        self.y_test_encoded = label_encoder.transform(self.y_test)
        self.num_classes = len(label_encoder.classes_)
        self.y_train_tensor = torch.tensor(self.y_train_encoded)
        self.y_test_tensor = torch.tensor(self.y_test_encoded)


def get_train_test_data(self, test_size=0.2):
        # Split the dataset into training and testing sets
        # ...
        return X_train, X_test, y_train, y_test