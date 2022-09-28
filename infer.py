
import pandas as pd
import torch
import numpy as np
from torch import nn
from tqdm import tqdm

from bert_models import bert, roberta, deberta
from Dataset_class import Dataset

import matplotlib.pyplot as plt
import seaborn as sns


#valid
def evaluate(model, bert_model ,test_data):
    prediction = []
    labels = []
    test =  Dataset(test_data, bert_model)
    
    test_dataloader = torch.utils.data.DataLoader(test,batch_size=2)
    criterion = nn.MSELoss()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    
    if use_cuda:
        
        model = model.cuda()
    
    
    total_loss_test = 0
    with torch.no_grad():
        for test_input, test_labels in tqdm(test_dataloader):
            test_labels =  test_labels.to(device)
            mask = test_input["attention_mask"].to(device)
            input_id = test_input["input_ids"].squeeze(1).to(device)
            
            output =  model(input_id, mask)
            
            for label in test_labels.cpu():
                labels.append(np.array([min(max(1.0, i), 5.0) for i in np.array(label)]))
             
            for pred in output.cpu():
                prediction.append(np.array([min(max(1.0, i), 5.0) for i in np.array(pred)]))

            loss =  criterion(output,test_labels)
            total_loss_test += loss
            
            
    print(f'Test_Loss: {total_loss_test / len(test_data): .3f}')
    return np.array(labels), np.array(prediction)
    # print(output_list[-1])


if __name__ == "__main__":

    # Loda dataset
    df = pd.read_csv("dataset/train.csv")

    np.random.seed(42)

    df_train, df_val, df_test = np.split(df.sample(frac=1,random_state=42),
                                        [int(.9*len(df)),
                                        int(.95*len(df))])

    #Run training
    
    # EPOCHS = 25
    # bert / roberta / deberta

    # bert_model = bert
    # model = torch.load("./trained_model/model_bert.pt")
   
    # bert_model = roberta
    # model = torch.load("./trained_model/model_roberta.pt")

    bert_model = deberta
    model = torch.load("./trained_model/model_deberta.pt")



    #evaluate
    labels, prediction = evaluate(model, bert_model, df_test)


    # visualization
    column_names = ["cohesion","syntax","vocabulary","phraseology","grammar","conventions" ]
    labels_data = pd.DataFrame(labels, columns= column_names)
    prediction_data = pd.DataFrame(prediction, columns= column_names)


    fig, axes = plt.subplots(
        nrows=1, ncols=6,
        sharey=True
        )
    axes[0].set_ylabel('Score')
    for i, measure in enumerate(column_names):
        
        axes[i].set_title(measure, fontsize = 8)
        axes[i].boxplot( [prediction_data[measure],labels_data[measure] ])
        axes[i].set_xticklabels(['pred', 'labels'], fontsize = 9)
        axes[i].set_ylim(0, 6)
    # plt.show()

    # plt.savefig("./model_plots/bert_model.png")
    # plt.savefig("./model_plots/roberta_model.png")
    plt.savefig("./model_plots/deberta_model.png")


