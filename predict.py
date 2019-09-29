# -*- coding: utf-8 -*
import pickle
import re
import tensorflow as tf
from bilstm_crf import Model


class NER_predict(object):

    def get_config(self):
        with open('../data/renmindata.pkl', 'rb') as inp:
            word2id = pickle.load(inp)
            id2word = pickle.load(inp)
            tag2id = pickle.load(inp)
            id2tag = pickle.load(inp)
            x_train = pickle.load(inp)
            y_train = pickle.load(inp)
            x_test = pickle.load(inp)
            y_test = pickle.load(inp)
            x_valid = pickle.load(inp)
            y_valid = pickle.load(inp)

        batch_size = 32
        config = {}
        config["lr"] = 0.001
        config["embedding_dim"] = 100
        config["sen_len"] = len(x_train[0])
        config["batch_size"] = batch_size
        config["embedding_size"] = len(word2id) + 1
        config["tag_size"] = len(tag2id)
        config["pretrained"] = False
        return word2id, id2tag, config


    def get_entity(self, x, y, id2tag,max_len):
        entity = ""
        res = []
        for i in range(len(x)):       # for every sen
            for j in range(max_len):  # for every word
                if y[i][j] == 0:
                    continue
                if id2tag[y[i][j]]:
                    if id2tag[y[i][j]][0] == 'B':
                        entity = id2tag[y[i][j]][2:] + ':' + x[i][j]
                    elif id2tag[y[i][j]][0] == 'M' and len(entity) != 0:
                        entity += x[i][j]
                    elif id2tag[y[i][j]][0] == 'E' and len(entity) != 0:
                        entity += x[i][j]
                        res.append(entity)
                        entity = []
                    else:
                        entity = []
                else:
                    entity = []
        return res


    def padding(self, ids,max_len):
        if len(ids) >= max_len:
            return ids[:max_len]
        else:
            ids.extend([0]*(max_len-len(ids)))
            return ids


    def test_input(self, text_part, model, sess, word2id, id2tag, batch_size, max_len=60):
        text_id = []
        for sen in text_part:
            word_id = []
            for word in sen:
                if word in word2id:
                    word_id.append(word2id[word])
                else:
                    word_id.append(word2id["unknow"])
            text_id.append(self.padding(word_id,max_len))

        zero_padding = []
        zero_padding.extend([0] * max_len)
        text_id.extend([zero_padding] * (batch_size - len(text_id)))
        feed_dict = {model.input_data: text_id}
        pre = sess.run([model.viterbi_sequence], feed_dict)
        entity = self.get_entity(text_part, pre[0], id2tag,max_len)
        return entity


    def predict_ner(self,text):
        word2id, id2tag, config = self.get_config()
        embedding_pre = []
        model = Model(config, embedding_pre, dropout_keep=1)
        batch_size = config['batch_size']
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state('./model')
            if ckpt is None:
                print('Model not found, please train your model first')
            else:
                path = ckpt.model_checkpoint_path
                print('loading pre-trained model from %s.....' % path)
                saver.restore(sess, path)
                text_list = re.split(u'[，。！？、‘’“”（）,.!?''"()]', text)
                if '' in text_list:
                    text_list.remove('')
                half_batch = batch_size // 8
                all_entity = []
                for i in range(0, len(text_list), half_batch):
                    text_part = text_list[i:i + half_batch]
                    entity = self.test_input(text_part, model, sess, word2id, id2tag, batch_size)
                    all_entity.extend(entity)
                new_entity = []
                for entity in all_entity:
                    entity_label = {}
                    entity = entity.split(':')
                    if entity[0] == 'nr':
                        entity[0] = '人名'
                    elif entity[0] == 'ns':
                        entity[0] = '地名'
                    elif entity[0] == 'nt':
                        entity[0] = '机构团体'
                    entity_label[entity[1]] = entity[0]
                    new_entity.append(entity_label)
                return new_entity


if __name__=='__main__':
    pass

    with open('./text.txt','r',encoding='utf-8') as file:
        text = file.read()

    ner = NER_predict()
    all_entity = ner.predict_ner(text)
    print(all_entity)
