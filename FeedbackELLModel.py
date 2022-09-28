import torch
import torch.nn as nn
from bert_models import *

#Create Model
class FeedbackELLModel(nn.Module):
    
    def __init__(self, bert_model, dropout=0.1):
        
        super(FeedbackELLModel, self).__init__()
        self.bert_model = bert_model

        self.bert = self.bert_model.model
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(1024,256)
        self.relu = nn.ReLU()
        self.out = nn.Linear(256,6)
        
    # bert / roberta
    # def forward(self,input_id,mask):
    #     _, x = self.bert(input_ids=input_id,attention_mask=mask,return_dict=False)

    #     x = self.dropout(x)
    #     x = self.linear(x)
    #     x = self.relu(x)
    #     final_layer = self.out(x)
    #     return final_layer
    
    # deberta
    def forward(self,input_id,mask):
        bert_out = self.bert(input_ids=input_id, attention_mask=mask)
        x = bert_out.last_hidden_state[:, 0, :]
        x = self.dropout(x)
        x = self.linear(x)
        x = self.relu(x)
        final_layer = self.out(x)
        return final_layer
        
        