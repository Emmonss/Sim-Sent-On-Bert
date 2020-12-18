import os
os.environ["TF_KERAS"]="1"

from config import config
from BERT_For_SimSent import BertModel_for_Simsent
from sklearn import metrics
from utils.log_utils import Logger
from itertools import chain
from BERT_Tokenize import tokenize_data,tokenize_data_2,tokenize_data_3
from Load_Data import load_data
import tensorflow as tf
import argparse

# gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
# assert len(gpu) == 1
# tf.config.experimental.set_memory_growth(gpu[0], True)

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
    train_token_ids, train_seg_ids, train_tags = tokenize_data_2(train_data)

    val_token_ids,val_seg_ids, val_tags = tokenize_data_2(val_data)
    logger.info("数据集一共有{}个训练数据，{}个验证数据".format(len(train_tags),len(val_tags)))

    sim_model = BertModel_for_Simsent()
    sim_model.model.summary()

    train_X = [train_token_ids, train_seg_ids]
    train_Y = (train_tags)
    val_X = [val_token_ids,val_seg_ids]
    val_Y = (val_tags)
    logger.info("training.....")
    sim_model.fit(train_X,train_Y,
                  valid_data=((val_X,val_Y)),
                  epochs=config.epoch,
                  batch_size=config.batch_size)

    if(config.is_save):
        logger.info('saving...')
        if not os.path.exists(config.model_save_path):
            os.mkdir(config.model_save_path)
        sim_model.save(config.model_save_path,config.model_name)

#加载已有的模型继续训练
def train_2(config):
    logger.info("正在加载数据集....")
    train_data, val_data = load_data(data_dir=config.data_dir, test_flag=False)

    logger.info("正在进行tokenize.....")
    train_token_ids, train_seg_ids, train_tags = tokenize_data_2(train_data)

    val_token_ids, val_seg_ids, val_tags = tokenize_data_2(val_data)
    logger.info("数据集一共有{}个训练数据，{}个验证数据".format(len(train_tags), len(val_tags)))

    train_X = [train_token_ids, train_seg_ids]
    train_Y = (train_tags)
    val_X = [val_token_ids, val_seg_ids]
    val_Y = (val_tags)

    model_path = os.path.join(os.path.join(config.model_save_path,config.model_name))
    mymodel = BertModel_for_Simsent()
    logger.info("加载模型{}".format(model_path))
    mymodel.model = tf.keras.models.load_model('{}.h5'.format(model_path))

    logger.info("training.....")
    mymodel.fit(train_X, train_Y,
                  valid_data=((val_X, val_Y)),
                  epochs=config.epoch,
                  batch_size=config.batch_size)

    if (config.is_save):
        logger.info('saving...')
        if not os.path.exists(config.model_save_path):
            os.mkdir(config.model_save_path)
        mymodel.save(config.model_save_path, config.model_name)


def infer(config,text1,text2):
    model_path = os.path.join(os.path.join(config.model_save_path, config.model_name))
    mymodel = BertModel_for_Simsent()
    logger.info("加载模型{}".format(model_path))
    mymodel.model = tf.keras.models.load_model('{}.h5'.format(model_path))
    print(type(mymodel.model))
    token_ids, seg_ids = tokenize_data_3([[text1,text2]])
    res = mymodel.predict([token_ids, seg_ids])
    print(res)
    pass

if __name__ == '__main__':
    text1 = "不要借了我是试试看能否操作的"
    text2 = "借款审核期间能否取消借款"
    infer(config=config,text1=text1,text2=text2)
# loss: 0.7487 - acc: 0.6110 - val_loss: 0.5904 - val_acc: 0.6990