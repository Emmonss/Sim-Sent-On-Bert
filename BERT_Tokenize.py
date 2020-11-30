from bert4keras.tokenizers import Tokenizer
from config import config
from sklearn.preprocessing import LabelEncoder
import os
import tensorflow as tf
import numpy as np
from Load_Data import load_data
from tqdm import tqdm



tokenizer = Tokenizer(os.path.join(config.model_dir,'vocab.txt'),do_lower_case=True)

def _tokenize(text):
    token_id, seg_id = tokenizer.encode("你是傻逼")
    return token_id,seg_id


def _pad_seuqences(tokens):
    return tf.keras.preprocessing.sequence.pad_sequences(tokens,
                                                         maxlen=config.seq_maxlen,
                                                         truncating='post',
                                                         padding='post')

def tokenize_data(data):
    token_ids_1 = []
    token_ids_2 = []
    seg_ids_1 = []
    seg_ids_2 = []
    tags = []
    for sent in tqdm(data):
        #token
        token_id_1, seg_id_1 = _tokenize(sent[0])
        token_id_2, seg_id_2 = _tokenize(sent[1])


        #append
        token_ids_1.append(token_id_1)
        token_ids_2.append(token_id_2)
        seg_ids_1.append(seg_id_1)
        seg_ids_2.append(seg_id_2)

        #target
        tags.append(str(sent[2]))
    #pad
    token_ids_1 = _pad_seuqences(token_ids_1)
    token_ids_2 = _pad_seuqences(token_ids_2)
    seg_ids_1 = _pad_seuqences(seg_ids_1)
    seg_ids_2 = _pad_seuqences(seg_ids_2)
    return token_ids_1,token_ids_2,seg_ids_1,seg_ids_2,tags







if __name__ == '__main__':
    train_data,val_data = load_data(data_dir='./data/bq_corpus')
    token_ids_1,token_ids_2,seg_ids_1,seg_ids_2,tags = tokenize_data(val_data)
    print(tags)
    pass