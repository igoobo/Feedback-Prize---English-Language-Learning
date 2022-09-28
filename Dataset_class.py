import torch
import numpy as np
import pandas as pd

from bert_models import *

#DatasetClass
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self,df, bert_model):
        
        self.labels = df[["cohesion","syntax","vocabulary","phraseology","grammar","conventions"]].reset_index()
        self.texts = df[["full_text"]].reset_index()
        self.bert_model = bert_model
    
    def classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)
    
    def get_batch_labels(self, idx):
        #fetch a batch of labels
        return np.array(self.labels.loc[idx].values[1:].astype(float))
        
        
    def get_batch_texts(self,idx):
        #fetch a batch of inputs
        if self.bert_model == deberta:
            max_length = 1024
        else:
            max_length = 512
            
        return self.bert_model.tokenizer(self.texts.loc[idx].values[1],
                        padding= 'max_length',max_length=max_length,truncation=True,
                        return_tensors='pt')
    
    def __getitem__(self,idx):
        
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        
        return batch_texts, batch_y
