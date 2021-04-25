

def openfile(filename):
    res= []
    with open(filename,'r',encoding='utf-8') as fr:
        for line in fr:
            res.append(line.strip().split('\t'))
            break
    return res


if __name__ == '__main__':
    import tensorflow as tf
    # import torch
    import os

    # os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

    a = tf.constant(1.)
    b = tf.constant(2.)
    print(a + b)

    print('GPU:', tf.test.is_gpu_available())
    # print('GPU:', torch.cuda.is_available())