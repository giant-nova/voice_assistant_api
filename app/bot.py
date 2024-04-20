import numpy as np
import json
import pandas as pd
import torch
import random
import torch.nn as nn
import gc
from transformers import BertTokenizer, BertModel
from torch.optim import Adam
from tqdm import tqdm
import pickle

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

with open('Dataset/Intent.json') as f:
    data = json.load(f)

class cfg:
    num_classes = 22
    epochs = 15
    batch_size = 10
    lr = 1e-5
    max_length = 15

df = pd.DataFrame(data['intents'])
df_patterns = df[['text', 'intent']]
df_responses = df[['responses', 'intent']]
df_patterns = df_patterns.explode('text')
df = df_patterns.copy()


labels={}
# if __name__ == "__main__":
s=set(list(df['intent']))
j=0
for i in s:
    labels[i]=j
    j+=1
    
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels = [labels[label] for label in df['intent']]
        self.texts = [tokenizer(text, padding='max_length', max_length=cfg.max_length, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.2):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, cfg.num_classes)

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        return linear_output

    def predict(self, text):
        text_dict = tokenizer(text, padding='max_length', max_length=cfg.max_length, truncation=True,
                              return_tensors="pt")
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        # Ensure that the model is on the selected device
        self.to(device)

        mask = text_dict['attention_mask'].to(device)
        input_id = text_dict['input_ids'].squeeze(1).to(device)

        with torch.no_grad():
            output = self(input_id, mask)
            label_id = output.argmax(dim=1)
            return label_id

df_train, df_val = np.split(df.sample(frac=1, random_state=42), [int(.9 * len(df))])

def train(model, train_data, val_data, learning_rate, epochs):
    train_dataset, val_dataset = Dataset(train_data), Dataset(val_data)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    model.to(device)

    for epoch_num in range(epochs):
        total_acc_train, total_loss_train = 0, 0

        model.train()
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val, total_loss_val = 0, 0
        model.eval()
        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data):.3f} | '
              f'Train Accuracy: {total_acc_train / len(train_data):.3f} | '
              f'Val Loss: {total_loss_val / len(val_data):.3f} | '
              f'Val Accuracy: {total_acc_val / len(val_data):.3f}')

    gc.collect()

model = BertClassifier()

EPOCHS = cfg.epochs
LR = cfg.lr
# train(model, df_train, df_val, LR, EPOCHS)

def process(message):
    message = str(message)
    # model.eval()
    prediction=model.predict(message)
    for i in labels:
        if labels[i]==prediction:
            # print(f"The intent of the given text is {i}")
            intent = i
    # print(intent)
    for i in range(len(data['intents'])):
        if data['intents'][i]['intent'] == intent:
            return str(data['intents'][i]['responses'][random.randint(0,len(data['intents'][i]['responses'])-1)])

if __name__ == "__main__":
    
    pickle.dump(model, open('Model/classifier.pkl', 'wb'))