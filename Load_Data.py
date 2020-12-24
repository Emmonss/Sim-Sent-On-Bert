import numpy as np
import os


def read_data(data_path,test_flag = False):
    data_list = []
    with open(data_path,'r',encoding='utf-8') as fr:
        for line in fr:
            out = line.strip().split('\t')
            if not test_flag:
                if len(out)==3:
                    data_list.append(out)
            else:
                data_list.append(out)
    return data_list



def load_data(data_dir,test_flag = False):
    train_data = read_data(os.path.join(data_dir,'train.tsv'),test_flag)
    val_data = read_data(os.path.join(data_dir,'dev.tsv'),test_flag)
    test_data = read_data(os.path.join(data_dir, 'test.tsv'),test_flag)

    if test_flag:
        del train_data,val_data
        return test_data
    else:
        return train_data,val_data


if __name__ == '__main__':
    load_data(data_dir='./data/bq_corpus')