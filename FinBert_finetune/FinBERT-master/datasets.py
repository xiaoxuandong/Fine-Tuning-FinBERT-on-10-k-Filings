import os 
import json 
import numpy as np 
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import pandas as pd

class text_dataset(Dataset):
    def __init__(self, x_y_list, vocab_path, max_seq_length=256, vocab = 'base-cased', transform=None):
        self.max_seq_length = max_seq_length
        self.x_y_list = x_y_list
        self.vocab = vocab
        if self.vocab == 'base-cased':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, do_basic_tokenize=True)
        elif self.vocab == 'finance-cased':
            self.tokenizer = BertTokenizer(vocab_file = vocab_path, do_lower_case = False, do_basic_tokenize = True)
        elif self.vocab == 'base-uncased':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True) 
        elif self.vocab == 'finance-uncased':
            self.tokenizer = BertTokenizer(vocab_file = vocab_path, do_lower_case = True, do_basic_tokenize = True)
    
    def __getitem__(self,index):
        tokenized_review = self.tokenizer.tokenize(self.x_y_list[0][index])
        
        if len(tokenized_review) > self.max_seq_length:
            tokenized_review = tokenized_review[:self.max_seq_length]
            
        ids_review  = self.tokenizer.convert_tokens_to_ids(tokenized_review)
        
        mask_input = [1]*len(ids_review)
        
        padding = [0] * (self.max_seq_length - len(ids_review))
        ids_review += padding
        mask_input += padding
        
        input_type = [0]*self.max_seq_length  
        
        assert len(ids_review) == self.max_seq_length
        assert len(mask_input) == self.max_seq_length
        assert len(input_type) == self.max_seq_length 
        
        ids_review = torch.tensor(ids_review)
        mask_input =  torch.tensor(mask_input)
        input_type = torch.tensor(input_type)
        
        sentiment = self.x_y_list[1][index] 
        list_of_labels = [torch.from_numpy(np.array(sentiment))]
        
        input_feature = {"token_type_ids": input_type, "attention_mask":mask_input, "input_ids":ids_review}
        
        return input_feature, list_of_labels[0]
    
    def __len__(self):
        return len(self.x_y_list[0])


def transform_labels(x_y_list):
    dict_labels = {'positive': 0, 'neutral':1, 'negative':2}
    x_y_list_transformed = [[item[0], dict_labels[item[1]]] for item in x_y_list]
    X = np.asarray([item[0] for item in x_y_list_transformed])
    y = np.asarray([item[1] for item in x_y_list_transformed])
    return X, y

def financialPhraseBankDataset(dir_):
    fb_path = os.path.join(dir_, 'FinancialPhraseBank-v1.0')
    data_50 = os.path.join(fb_path, 'Sentences_50Agree.txt')
    sent_50 = []
    rand_idx = 45
    
    with open(data_50, 'rb') as fi:
        for l in fi:
            l = l.decode('utf-8', 'replace')
            sent_50.append(l.strip())
    
    x_y_list_50 = [sent.split("@") for sent in sent_50]
    x50, y50 = transform_labels(x_y_list_50)
    
    data = [x50, y50]
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.1, random_state=rand_idx, stratify=data[1])

    y_train = pd.get_dummies(y_train).values.tolist()
    y_test = pd.get_dummies(y_test).values.tolist()
    X_train = X_train.tolist()
    X_test = X_test.tolist()
            
    final_data = [X_train, X_test, y_train, y_test] 
     
    return final_data


def ourOwnDataset(dir_):
    fb_path = os.path.join(dir_, 'dataset')
    data_5000 = os.path.join(fb_path, 'ebay_label.csv')
    test_500 = os.path.join(fb_path, 'testset.csv')
    rand_idx = 45
    df = pd.read_csv(data_5000)
    df_test = pd.read_csv(test_500)

    df1 = df[['headline','manually label here']]
    df_test1 = df_test[['sentence','label']]
    # df1.reset_index(inplace=True, drop=True)
    # print(df1)
    # df_test1.reset_index(inplace=True, drop=True)
    x5000=df1['headline'].tolist()
    y5000=df1['manually label here'].tolist()
    # print(y5000)
    # print(np.isnan(y5000).any())








                   
    # df1 = df[['headline','sentiment mapped']][:]
    # # df_test1 = df_test[['headline','sentiment mapped']]
    # df1.reset_index(inplace=True, drop=True)
    # # df_test1.reset_index(inplace=True, drop=True)
    # x5000=df1['headline'].tolist()
    # y5000=df1['sentiment mapped'].tolist()
    x500=df_test1['sentence'].tolist()
    y500=df_test1['label'].tolist()
    data = [x5000, y5000]
    # print(x5000)

    # data_test = [x500, y500]

    # other test dataset
    # X_train=x5000
    # X_test=x500
    # y_train=y5000
    # y_test=y500

    #split test dataset
    X_train, X_val, y_train, y_val = train_test_split(data[0], data[1], test_size=0.1, random_state=rand_idx, stratify=data[1])
    X_test = x500
    y_test = y500
    # print(y_test)
    c={'trainset':X_train,'label':y_train}
    d={'valset':X_val,'label':y_val}
    dfc=pd.DataFrame(c)
    dfd=pd.DataFrame(d)
    dfc.to_csv('./output/train_label.csv',index=False)
    dfd.to_csv('./output/val_label.csv',index=False)
    y_train = pd.get_dummies(y_train).values.tolist()
    y_val = pd.get_dummies(y_val).values.tolist()
    y_test = pd.get_dummies(y_test).values.tolist()
    print(len(y_test))
    # X_train = X_train.tolist()
    # X_test = X_test.tolist()
            
    final_data = [X_train, X_val, X_test, y_train, y_val, y_test]
    return final_data
