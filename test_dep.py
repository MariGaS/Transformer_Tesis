import csv
import itertools
import numpy as np
import os
import pandas as pd
import random
import torch
import tqdm
import transformers as ppb

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from torch import optim

from transformers import AutoModel, AutoTokenizer, BertForMaskedLM, BertTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score



def padding_chunk(input_id_chunks, mask_chunks):
    # get required padding length
    pad_len = 128 - input_id_chunks.shape[0]
    # check if tensor length satisfies required chunk size
    if pad_len > 0:
        # if padding length is more than 0, we must add padding
        input_id_chunks = torch.cat([input_id_chunks, torch.Tensor([0] * pad_len)])
        mask_chunks = torch.cat([mask_chunks, torch.Tensor([0] * pad_len)])
    
    return input_id_chunks, mask_chunks

# Helper functions
def to_int(x):
    return list(map(int,x))

def to_low(x):
    return [w.lower() for w in x]

def test_model(model, test_batches, _config):
    device = torch.device(_config['device'])
    y_t=[]
    y_p=[]
    logits = []
    for sample in test_batches:
        if sample['text'].size(dim = 1)> 126: 
            cont_1 =0 
            cont_0 = 0
            input_id_chunks = list(sample['text'][0].split(126))
            mask_chunks =  list(sample['attention'][0].split(126))

            # loop through each chunk
            for i in range(len(input_id_chunks)):
                input_id_chunks[i] = torch.cat([torch.tensor([101]), input_id_chunks[i], torch.tensor([102])])
                # add attention tokens to attention mask
                mask_chunks[i] = torch.cat([torch.tensor([1]), mask_chunks[i], torch.tensor([1])])
                input_id_chunks[i], mask_chunks[i] = padding_chunk(input_id_chunks[i], mask_chunks[i])
                        
                        
                x, att, y = input_id_chunks[i][None, :], mask_chunks[i][None, :], sample['label']
                x, att = x.to(torch.int64), att.to(torch.int64)

                x, y, att = x.to(device), y.to(device), att.to(device)
                y_pred = F.softmax(model(x, att).cpu().detach(),1)
                y_pred = y_pred.argmax(1)
                if y_pred.item() == 1:
                    cont_1 += 1
                else: 
                    cont_0 += 1
            if cont_1 > cont_0: 
                y_pred = torch.tensor([1])
            else: 
                y_pred = torch.tensor([0])
            y_p.append(y_pred)
            y_t.append(y.cpu())
            
        else:
            x, y, att = sample['text'][0], sample['label'], sample['attention'][0]
            x= torch.cat([torch.tensor([101]), x, torch.tensor([102])])
            att = torch.cat([torch.tensor([1]), att, torch.tensor([1])])
            x, att = padding_chunk(x, att)
            x, att = x[None,:], att[None, :]
            x, att = x.to(torch.int64), att.to(torch.int64)
            x, y, att = x.to(device), y.to(device), att.to(device)
            #x, y, att = sample['text'].to(device), sample['label'].to(device), sample['attention'].to(device)
            y_pred = F.softmax(model(x, att).cpu().detach(),1)
            logits.append(y_pred)
            y_pred = y_pred.argmax(1)
            y_p.append(y_pred)
            y_t.append(y.cpu())
    logits = torch.cat(logits)
    y_p=torch.cat(y_p)
    y_t=torch.cat(y_t)
    return f1_score(y_t,y_p,average='binary')

def get_logits(model, test_batches, _config):
    device = torch.device(_config['device'])
    logits = []
    count = 0
    for sample in test_batches:
        
        if sample['text'].size(dim = 1)> 126: 
            
            list_tensors = []
            input_id_chunks = list(sample['text'][0].split(126))
            mask_chunks =  list(sample['attention'][0].split(126))

            # loop through each chunk
            for i in range(len(input_id_chunks)):
                # add CLS and SEP tokens to input IDs
#                 if i == 0:
#                     input_id_chunks[i] = torch.cat([input_id_chunks[i], torch.tensor([102])])
#                     # add attention tokens to attention mask
#                     mask_chunks[i] = torch.cat([mask_chunks[i], torch.tensor([1])])
#                 if i == len(input_id_chunks)-1: 
#                     input_id_chunks[i] = torch.cat([torch.tensor([101]), input_id_chunks[i]])
#                     # add attention tokens to attention mask
#                     mask_chunks[i] = torch.cat([torch.tensor([1]),mask_chunks[i]])
#                     input_id_chunks[i], mask_chunks[i] = padding_chunk(input_id_chunks[i], mask_chunks[i])

                input_id_chunks[i] = torch.cat([torch.tensor([101]), input_id_chunks[i], torch.tensor([102])])
                # add attention tokens to attention mask
                mask_chunks[i] = torch.cat([torch.tensor([1]), mask_chunks[i], torch.tensor([1])])
                input_id_chunks[i], mask_chunks[i] = padding_chunk(input_id_chunks[i], mask_chunks[i])
                        
                        
                x, att, y = input_id_chunks[i][None, :], mask_chunks[i][None, :], sample['label']
                x, att = x.to(torch.int64), att.to(torch.int64)

                x, y, att = x.to(device), y.to(device), att.to(device)
                y_pred = F.softmax(model(x, att).cpu().detach(),1)
                list_tensors.append(y_pred)
            
            my_tensor = torch.cat(list_tensors, dim=0)
            
            logit_app = torch.mean(my_tensor, 0, dtype= float)
            logit_app = logit_app[None,:]
            logits.append(logit_app)
        else:
            x, y, att = sample['text'][0], sample['label'], sample['attention'][0]
            x= torch.cat([torch.tensor([101]), x, torch.tensor([102])])
            att = torch.cat([torch.tensor([1]), att, torch.tensor([1])])
            x, att = padding_chunk(x, att)
            x, att = x[None,:], att[None, :]
            x, att = x.to(torch.int64), att.to(torch.int64)
            x, y, att = x.to(device), y.to(device), att.to(device)
            #x, y, att = sample['text'].to(device), sample['label'].to(device), sample['attention'].to(device)
            y_pred = F.softmax(model(x, att).cpu().detach(),1)
            logits.append(y_pred)
    logits = torch.cat(logits)
    return logits


# Define dataloader
class Mental(Dataset):
    def __init__(self, X,y = None, _config = None):
        self.X = X
        self.y = y
        self.tokenizer = BertTokenizer.from_pretrained(_config['model_name'], do_lower_case=True)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        out=torch.tensor(self.tokenizer.encode(self.X[idx], max_length=128, pad_to_max_length=True, add_special_tokens=True))
        if(self.y):
            return {"text": out, "attention":(out!=1).float(), "label":torch.tensor(self.y[idx])}
        else:
            return {"text": out, "attention":(out!=1).float()}

class TestMental(Dataset):
    def __init__(self, X,y = None, _config = None):
        self.X = X
        self.y = y
        self.tokenizer = BertTokenizer.from_pretrained(_config['model_name'], do_lower_case=True)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        out=torch.tensor(self.tokenizer.encode(self.X[idx], add_special_tokens=False))
        if(self.y):
            return {"text": out, "attention":(out!=1).float(), "label":torch.tensor(self.y[idx])}
        else:
            return {"text": out, "attention":(out!=1).float()}

# Define model
class BERTForSequenceClassification(nn.Module):
    def __init__(self, model_name):
        super(BERTForSequenceClassification, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
        self.drop = nn.Dropout(0.1)
        self.clf  = nn.Linear(768, 2, bias=True)
    def forward(self, x, att):
        x = self.bert(x, attention_mask = att)[1]
        x = self.drop(x)
        x = self.clf(x)
        return x    

def train_models(_config, train, test, verbose = True):
    device = torch.device(_config['device'])
    w = _config['weights']
    lr = _config['lr']
    max_grad_norm = 1.0
    epochs = _config['epochs']
    for k in range(0, _config['n_models']):
        train_batches = DataLoader(train, batch_size = _config['train_batch_size'], shuffle = True)
        test_batches = DataLoader(test, batch_size = _config['test_batch_size'], shuffle = False)
        
        model = BERTForSequenceClassification(_config['model_name']).to(device)
        optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)
        total_steps = len(train_batches) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
        criterio = nn.CrossEntropyLoss(weight = w.to(device))

        for epoch in range(epochs):
            for sample in train_batches:
                optimizer.zero_grad()
                x, y, att = sample['text'].to(device), sample['label'].to(device), sample['attention'].to(device)
                y_pred = model(x, att)
                loss = criterio(y_pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

        if(verbose):
            model.eval()
            tmp_m = test_model(model, test_batches, _config)
            
            print("Model %d \t f_1 = %.4f"%(k, tmp_m*100))
        
        torch.save(model.state_dict(), '/home/est_posgrado_maria.garcia/Transformers/Bert_base_uncased/BERT/Models_dep/model_'+str(k)+'.pt')

Df_dep_train_128 = pd.read_csv('/home/est_posgrado_maria.garcia/Transformers/Bert_base_uncased/BERT/Data_post/train_128_dep_clean.csv')
Df_dep_train_128['text'] = Df_dep_train_128['text'].astype('str')
X_train = to_low(list(Df_dep_train_128['text']))
y_train = to_int(list(Df_dep_train_128['target']))
df_test = pd.read_csv('/home/est_posgrado_maria.garcia/Transformers/Bert_base_uncased/BERT/Data_post/test_dep_clean.csv')
# Prepare test
X_test = to_low(list(df_test['text']))
y_test = to_int(list(df_test['target']))


w = (lambda a, b: torch.tensor([max(a, b)/a, max(a,b)/b]))((torch.tensor(y_train)==0).sum().float(), (torch.tensor(y_train)==1).sum().float())

_config = {
    'model_name': 'bert-base-uncased',
    'train_batch_size':  128,
    'test_batch_size': 1,
    'device': 'cuda:0',
    'lr': 1e-5,
    'epochs': 3,
    'n_models': 3,
    'weights': w
}

train = Mental(X_train, y_train, _config) 
test = TestMental(X_test, y_test, _config) 
_config['n_models'] = 3
train_models(_config, train, test)
test_batches = DataLoader(test, batch_size = _config['test_batch_size'], shuffle = False)
device = torch.device(_config['device'])
model = BERTForSequenceClassification(_config['model_name']).to(device)
x = []
for i in tqdm.tqdm(range(_config['n_models'])):
    model.load_state_dict(torch.load("/home/est_posgrado_maria.garcia/Transformers/Bert_base_uncased/BERT/Models_dep/model_"+str(i)+".pt"))
    model.eval()
    x.append(get_logits(model, test_batches, _config).unsqueeze(1))
    
X = torch.cat(x,1)


# Majority Voting 
y_pred = (X.argmax(2).sum(1).float()/_config['n_models']).round()
f1_score(y_test, y_pred)*100
print(f1_score(y_test, y_pred)*100)
# Weighted Voting 
y_pred= X.sum(1).argmax(1)
f1_score(y_test, y_pred)*100
print('Weighted Voting:', f1_score(y_test, y_pred)*100)