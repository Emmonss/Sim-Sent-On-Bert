import os
os.environ["TF_KERAS"]="1"

from config import config
from BERT_For_SimSent import BertModel_for_Simsent
from sklearn import metrics
from utils.log_utils import Logger
from itertools import chain
from BERT_Tokenize import tokenize_data
from Load_Data import load_data

import argparse
import pickle
import json
import numpy as np


logger = Logger("Main").get_logger()

def parse_args():
    parser = argparse.ArgumentParser("Trainning the Bert On Sim Sentences Discrimination")
    parser.add_argument("--mode",type=str,default='train',help='model of ["train","eval"]')
    args = parser.parse_args()
    return args

def flatten(y):
    return list(chain.from_iterable(y))

def train(config):
    logger.info("正在加载数据集....")
    train_data, val_data = load_data(data_dir=config.data_dir,test_flag=False)

    logger.info("正在进行tokenize.....")
    train_token_ids_1, train_token_ids_2, \
    train_seg_ids_1, train_seg_ids_2, train_tags = tokenize_data(train_data)

    val_token_ids_1, val_token_ids_2, \
    val_seg_ids_1, val_seg_ids_2, val_tags = tokenize_data(val_data)
    logger.info("数据集一共有{}个训练数据，{}个验证数据".format(len(train_tags),len(val_tags)))

    sim_model = BertModel_for_Simsent()
    sim_model.model.summary()
    print(np.shape(train_token_ids_1))
    train_X = [train_token_ids_1,train_seg_ids_1,train_token_ids_2,train_seg_ids_2]
    train_Y = [train_tags]
    val_X = [val_token_ids_1,val_seg_ids_1,val_token_ids_2,val_seg_ids_2]
    val_Y = [val_tags]

    logger.info("training.....")
    sim_model.fit(train_X,train_Y,
                  valid_data=(val_X,val_Y),
                  epochs=config.epoch,
                  batch_size=config.batch_size)

    if(config.is_save):
        logger.info('saving...')
        if not os.path.exists(config.model_save_path):
            os.mkdir(config.model_save_path)
        sim_model.save(config.model_save_path,config.model_name)

if __name__ == '__main__':
    train(config=config)
