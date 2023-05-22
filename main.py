import os
import pandas as pd
import numpy as np
from data.WhaleDataset import WhaleDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import torchvision
import torchvision.transforms as transforms


def oneHotEncode_labels(labels):
    ogValues = np.array(labels)
    label_encoder = LabelEncoder()
    int_encoded_values = label_encoder.fit_transform(ogValues)
    int_encoded_values = int_encoded_values.reshape(len(int_encoded_values), 1)
    oneHot_encoder = OneHotEncoder(sparse=False)

    oneHotEncoded_values = oneHot_encoder.fit_transform(int_encoded_values)

    return oneHotEncoded_values, label_encoder

def main():

    train_df = pd.read_csv("./data/humpback-whale-identification/train.csv")
    print(train_df.head)
    print(train_df.Id.value_counts().head())
    
    



if __name__ == "__main__":
  main()
