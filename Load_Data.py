import numpy as np
import os


def read_data(data_path):
    data_list = []
    with open(data_path,'r',encoding='utf-8') as fr:
        for line in fr:
            out = line.strip().split('\t')
            if len(out)==3:
                data_list.append(out)
    return data_list



def load_data(data_dir,test_flag = False):
    if test_flag:
        test_data = read_data(os.path.join(data_dir, 'test.tsv'))
        return test_data
    else:
        train_data = read_data(os.path.join(data_dir, 'train.tsv'))
        val_data = read_data(os.path.join(data_dir, 'dev.tsv'))
        return train_data,val_data


if __name__ == '__main__':
    load_data(data_dir='./data/bq_corpus')