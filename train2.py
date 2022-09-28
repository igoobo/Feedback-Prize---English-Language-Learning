import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.optim import lr_scheduler
from tqdm import tqdm

from bert_models import bert, roberta, deberta
from Dataset_class import Dataset
from FeedbackELLModel import FeedbackELLModel


#Training function
def train(model, bert_model, train_data, val_data, epochs):

    train, val = Dataset(train_data, bert_model), Dataset(val_data, bert_model)
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  
    device = torch.device(dev) 
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-5)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=500, 
                                                                 eta_min=1e-6)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_loss_train = 0

            for train_input, train_labels in tqdm(train_dataloader):

                train_labels = train_labels.to(device).float()
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                
                batch_loss = criterion(output, train_labels)
                total_loss_train += batch_loss.item()
                
                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
                scheduler.step()
            
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                                
            print(f"""Epoch: {epoch_num + 1} | 
            Train Loss: {total_loss_train / len(train_data): .3f} 
            | Val Loss: {total_loss_val / len(val_data): .3f}""")


if __name__ == "__main__":

    # Loda dataset
    df = pd.read_csv("dataset/train.csv")

    np.random.seed(42)

    df_train, df_val, df_test = np.split(df.sample(frac=1,random_state=42),
                                        [int(.9*len(df)),
                                        int(.95*len(df))])

    #Run training
    
    EPOCHS = 25
    # bert / roberta / deberta
    # bert_model = bert
    # model = FeedbackELLModel(bert_model) 
    # train(model, bert_model, df_train, df_val, EPOCHS)
    # torch.save(model, "./trained_model/model_bert.pt")


    # bert_model = roberta
    # model = FeedbackELLModel(bert_model) 
    # train(model, bert_model, df_train, df_val, EPOCHS)
    # torch.save(model, "./trained_model/model_roberta.pt")



    bert_model = deberta
    model = FeedbackELLModel(bert_model) 
    train(model, bert_model, df_train, df_val, EPOCHS)
    torch.save(model, "./trained_model/model_deberta.pt")


    # #evaluate
    # evaluate(model, df_test)