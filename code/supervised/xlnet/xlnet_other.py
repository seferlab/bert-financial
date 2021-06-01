# -*- coding: utf-8 -*-
"""
Created on Tue May 25 13:23:54 2021

@author: Erva
"""
import transformers
from transformers import XLNetTokenizer, XLNetModel, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import defaultdict
from textwrap import wrap
from pylab import rcParams

from torch import nn, optim
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset,RandomSampler,SequentialSampler
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import XLNetTokenizerFast
import pandas as pd
#train = pd.read_csv('/content/arrangedallagree.csv' ,encoding="latin",header=None)#, names=['No','text','labels'])
#train.columns = ["text",'labels']

train = pd.read_csv('/home/emresefer/zehra_old/supervised_files/berttrainfiqapost2.tsv', sep='\t', names=['No','text','labels'])
print("train: post!")
label_list = [-1,0,1] # 0,1,2
train.labels+=1
train = train[['text','labels']]
from transformers import XLNetForSequenceClassification
model = XLNetForSequenceClassification.from_pretrained("/home/emresefer/zehra_old/xlnet/200MB", num_labels = 3)

BATCH_SIZE = 1
MAX_LEN = 512
EPOCHS = 3
LEARNING_RATE = 1e-05
tokenizer = XLNetTokenizerFast('/home/emresefer/zehra_old/xlnet/200MB/sp10m.cased.v3.model', truncation=True)
from keras.preprocessing.sequence import pad_sequences


class ImdbDataset(Dataset):

    def __init__(self, text, labels, tokenizer, max_len):
        self.text = text
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, item):
        text = str(self.text[item])
        labels = self.labels[item]

        encoding = self.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True,
        )

        input_ids = pad_sequences(encoding['input_ids'], maxlen=MAX_LEN, dtype=torch.Tensor ,truncating="post",padding="post")
        input_ids = input_ids.astype(dtype = 'int64')
        input_ids = torch.tensor(input_ids) 

        attention_mask = pad_sequences(encoding['attention_mask'], maxlen=MAX_LEN, dtype=torch.Tensor ,truncating="post",padding="post")
        attention_mask = attention_mask.astype(dtype = 'int64')
        attention_mask = torch.tensor(attention_mask)       

        return {
        'review_text': text,
        'input_ids': input_ids,
        'attention_mask': attention_mask.flatten(),
        'targets': torch.tensor(labels, dtype=torch.long)
        }
    
def create_data_loader(df, tokenizer, max_len, batch_size):
 ds = ImdbDataset(
   text=df.text.to_numpy(),
   labels=df.labels.to_numpy(),
   tokenizer=tokenizer,
   max_len=max_len
 )

 return DataLoader(
   ds,
   batch_size=batch_size,
   num_workers=4
 )   
   
EPOCHS = 3
train_data_loader = create_data_loader(train, tokenizer, 512, 1)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
                                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

for i in range(0,5):
    if i==0:
        print("val is: headline\n")
        val = pd.read_csv('/home/emresefer/zehra_old/supervised_files/berttrainfiqaheadline2.tsv', sep='\t', names=['No','text','labels'])
        del val['No']
        val.labels +=1
        val.head()
    elif i==1:
        print("val is: aranged all\n")
        val = pd.read_csv('/home/emresefer/zehra_old/supervised_files/arrangedallagree.csv' ,encoding="latin",header=None)#, names=['No','text','labels'])
        val.columns = ["text",'labels']
        val.labels +=1
        val.head()
    elif i==2:
        print("val is: aranged 75\n")
        val = pd.read_csv('/home/emresefer/zehra_old/supervised_files/arranged75agree.csv' ,encoding="latin",header=None)#, names=['No','text','labels'])
        val.columns = ["text",'labels']
        val.labels +=1
        val.head()
    elif i==3:
        print("val is: aranged 66\n")
        val = pd.read_csv('/home/emresefer/zehra_old/supervised_files/arranged66agree.csv' ,encoding="latin",header=None)#, names=['No','text','labels'])
        val.columns = ["text",'labels']
        val.labels +=1
        val.head()
    elif i==4:
        print("val is: aranged 50\n")
        val = pd.read_csv('/home/emresefer/zehra_old/supervised_files/arranged50agree.csv' ,encoding="latin",header=None)#, names=['No','text','labels'])
        val.columns = ["text",'labels']
        val.labels +=1
        val.head()
    val_data_loader = create_data_loader(val, tokenizer, 512, 1)
    data = next(iter(val_data_loader))
    print(data.keys())
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    targets = data['targets']
    #print(input_ids.reshape(4,512).shape) # batch size x seq length
    #print(attention_mask.shape) # batch size x seq length
    outputs = model(input_ids.reshape(1,512), token_type_ids=None, attention_mask=attention_mask, labels=targets)
    print("beginning of train funcccc\n")
    from sklearn import metrics
    def train_epoch(model, data_loader, optimizer, scheduler, n_examples):
        model = model.train()
        losses = []
        acc = 0
        counter = 0
      
        for d in data_loader:
            input_ids = d["input_ids"].reshape(1,512)
            attention_mask = d["attention_mask"]
            targets = d["targets"]
            #print(input_ids.shape) # batch size x seq length
            #print(attention_mask.shape)
            outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels = targets)
            loss = outputs[0]
            logits = outputs[1]
    
            # preds = preds.cpu().detach().numpy()
            _, prediction = torch.max(outputs[1], dim=1)
            targets = targets.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()
            accuracy = metrics.accuracy_score(targets, prediction)
    
            acc += accuracy
            losses.append(loss.item())
            
            loss.backward()
    
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            counter = counter + 1
    
        return acc / counter, np.mean(losses)
    def eval_model(model, data_loader, n_examples):
        model = model.eval()
        losses = []
        acc = 0
        counter = 0
      
        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].reshape(1,512)
                attention_mask = d["attention_mask"]
                targets = d["targets"]
                
                outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels = targets)
                loss = outputs[0]
                logits = outputs[1]
    
                _, prediction = torch.max(outputs[1], dim=1)
                targets = targets.numpy()#cpu().detach().numpy()
                prediction = prediction.numpy()#cpu().detach().numpy()
                accuracy = metrics.accuracy_score(targets, prediction)
    
                acc += accuracy
                losses.append(loss.item())
                counter += 1
    
        return acc / counter, np.mean(losses)
    history = defaultdict(list)
    best_accuracy = 0
    
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
    
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,     
            optimizer, 
            scheduler, 
            len(train)
        )
    
        print(f'Train loss {train_loss} Train accuracy {train_acc}')
    
        val_acc, val_loss = eval_model(
            model,
            val_data_loader,  
            len(val)
        )
    
        print(f'Val loss {val_loss} Val accuracy {val_acc}')
        print()
    
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
    
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), '/home/emresefer/zehra_old/supervised_files/xlnet_model.bin')
            best_accuracy = val_acc
        model.load_state_dict(torch.load('/home/emresefer/zehra_old/supervised_files/xlnet_model.bin'))
        test_acc, test_loss = eval_model(
          model,
          val_data_loader,
          len(val)
        )
        
        print('Test Accuracy :', val_acc)
        print('Test Loss :', val_loss)
                




#test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)










