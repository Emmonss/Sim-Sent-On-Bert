data_dir = './data/bq_corpus'

model_dir = './models/chinese_L-12_H-768_A-12'

batch_size =100
epoch = 30
seq_maxlen = 256

bert_config = 'bert_config.json'
bert_ckpt = 'bert_model.ckpt'


model_save_path = './model_hub'
learning_rate = 5e-5
dropout=0.1
class_num = 2
class Config(object):
    def __init__(self, **kwargs):
        for key,value in kwargs.items():
            self.__setattr__(key,value)

    def add_argument(self,key,value):
        self.__setattr__(key, value)


config = Config(
    seq_maxlen = seq_maxlen,
    epoch = epoch,
    data_dir = data_dir,
    model_dir = model_dir,
    batch_size=batch_size,
    bert_config=bert_config,
    bert_ckpt=bert_ckpt,
    learning_rate = learning_rate,
    dropout=dropout,
    class_num=class_num,
    model_save_path=model_save_path
)

if __name__ == '__main__':
    print(config.data_dir)





