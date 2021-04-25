import os,shutil

root_path = './data'

merge_data = ['bq_corpus','lcqmc','paws-x-zh']
out_file_name = 'merge'

def get_lines(path):
    res = []
    with open(path,'r',encoding='utf-8') as fr:
        for line in fr:
            res.append(line)
    return res

def dumps_lines(path,line_list):
    with open(path,'a',encoding='utf-8') as fw:
        for item in line_list:
            fw.write(item)
    fw.close()

if __name__ == '__main__':
    out_dir = os.path.join(root_path,out_file_name)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    for item in merge_data:
        item_file = os.path.join(root_path, item)
        if os.path.exists(item_file):
            train_data_path = os.path.join(item_file,'train.tsv')
            dumps_lines(os.path.join(out_dir,'train.tsv'),get_lines(train_data_path))
            val_data_path = os.path.join(item_file, 'dev.tsv')
            dumps_lines(os.path.join(out_dir, 'dev.tsv'), get_lines(val_data_path))
