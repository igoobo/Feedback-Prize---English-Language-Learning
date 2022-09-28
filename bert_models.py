from transformers import BertTokenizer, BertModel
from transformers import RobertaModel ,RobertaTokenizer
from transformers import DebertaTokenizer, DebertaModel

class bert:
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertModel.from_pretrained('bert-large-uncased')

class roberta:
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    model = RobertaModel.from_pretrained("roberta-large")

class deberta:
    tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-large")
    model =  DebertaModel.from_pretrained("microsoft/deberta-large")


