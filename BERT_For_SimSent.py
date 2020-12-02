import os
os.environ["TF_KERAS"] = "1"
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input,Dense,Multiply,Dropout
from tensorflow.python.keras.layers.merge import concatenate
from NLUModel import NLUModel
from bert4keras.models import build_transformer_model
from config import config

import numpy as np
import os
import json


class BertModel_for_Simsent(NLUModel):
    def __init__(self):
        self.model=None
        self.bert_model_2()
        self.compile_model()


    def load_bert(self,ex_name = None):
        config_path = os.path.join(config.model_dir, config.bert_config)
        ckpt_path = os.path.join(config.model_dir, config.bert_ckpt)
        model =build_transformer_model(config_path=config_path,checkpoint_path=ckpt_path,model='bert')
        if not ex_name == None:
            for index in range(len(model.layers)):
                layer = model.get_layer(index=index)
                layer._name = '{}_{}'.format(layer._name,ex_name)
        return model

    def bert_model_2(self):
        bert = self.load_bert()
        seq, seg = bert.input

        bert_out = bert.output
        bert_sent = bert_out[:, 0, :]
        bert_sent_drop = Dropout(rate=config.dropout, name="bert_sent_drop")(bert_sent)

        sent_tc = Dense(config.class_num,activation='softmax',name='sim_classifier')(bert_sent_drop)
        self.model = Model(inputs=[seq, seg], outputs=[sent_tc])
        pass

    # def build_model(self):
    #     bert1 = self.load_bert(ex_name='sent_1')
    #     bert2 = self.load_bert(ex_name='sent_2')
    #
    #     seq1,seg1 = bert1.input
    #     seq2,seg2 = bert2.input
    #
    #     bert_out_1 = bert1.output
    #     bert_out_2 = bert2.output
    #
    #     bert_sent_1 = bert_out_1[:,0,:]
    #     bert_sent_2 = bert_out_2[:, 0, :]
    #
    #     bert_sent_drop_1 = Dropout(rate=config.dropout,name="bert_sent1_drop")(bert_sent_1)
    #     bert_sent_drop_2 = Dropout(rate=config.dropout, name="bert_sent2_drop")(bert_sent_2)
    #
    #     bert_sent_merge = concatenate([bert_sent_drop_1,bert_sent_drop_2], axis=1)
    #     bert_sent_merge = Dropout(rate=config.dropout, name="bert_sent_merge_drop")(bert_sent_merge)
    #
    #     fc = Dense(300,activation='relu',name='hidden_sim')(bert_sent_merge)
    #
    #     output = Dense(config.class_num,activation='softmax',name='sim_classifier')(fc)
    #
    #     self.model = Model(inputs=[seq1,seg1,seq2,seg2],outputs=[output])


    def compile_model(self):
        opt = tf.keras.optimizers.Adam(lr=config.learning_rate)
        loss = {
            'sim_classifier':'sparse_categorical_crossentropy'
        }
        loss_weight = {'sim_classifier':1.0}
        metrics = {'sim_classifier':'acc'}
        self.model.compile(optimizer=opt,loss=loss,loss_weight=loss_weight,metrics=metrics)

    def fit(self,X,Y,valid_data=None,epochs=6,batch_size=32):
        self.model.fit(X,Y,validation_data=valid_data,epochs=epochs,batch_size=batch_size)

    def save(self,model_path,model_name):
        self.model.save(os.path.join(model_path,'{}.h5'.format(model_name)))



if __name__ == '__main__':
    sim_model = BertModel_for_Simsent()
    sim_model.model.summary()
    pass