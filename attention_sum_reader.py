import logging
import os
import random
# noinspection PyUnresolvedReferences
import time

# noinspection PyPep8Naming
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import LambdaCallback, EarlyStopping
from keras.engine import Input, Model
from keras.layers import GRU, LSTM, Bidirectional, Embedding, Lambda, Activation, Multiply
from keras.optimizers import SGD, Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model


class AttentionSumReader(object):
    def __init__(self,
                 word_dict,
                 embedding_matrix,
                 d_len,
                 q_len,
                 embedding_dim,
                 hidden_size,
                 num_layers,
                 weight_path,
                 use_lstm=False):
        """
        初始化模型
        b ... batch_size
        t ... d_len
        f ... hidden_size*2
        i ... candidate_len 
        """
        self.weight_path = weight_path
        self.word_dict = word_dict
        self.vocab_size = len(embedding_matrix)
        self.d_len = d_len
        self.q_len = q_len
        self.A_len = 10

        logging.info("Embedding matrix shape:%d x %d" % (len(embedding_matrix), embedding_dim))

        self.rnn_cell = LSTM if use_lstm else GRU
        self.cell_name = "LSTM" if use_lstm else "GRU"

        # 模型的输入
        q_input = Input(batch_shape=(None, self.q_len), dtype="int32", name="q_input")
        d_input = Input(batch_shape=(None, self.d_len,), dtype="int32", name="d_input")
        context_mask = Input(batch_shape=(None, self.d_len), dtype="float32", name="context_mask")
        candidates_bi = Input(batch_shape=(None, self.A_len), dtype="int32", name="candidates_bi")

        # 问题的编码模型
        # output shape: (None, max_q_length, embedding_dim)
        q_encode = Embedding(input_dim=self.vocab_size,
                             output_dim=embedding_dim,
                             weights=[embedding_matrix],
                             mask_zero=True,
                             )(q_input)
        for i in range(1, num_layers):
            q_encode = Bidirectional(self.rnn_cell(units=hidden_size,
                                                   name="{}-{}-{}".format("q-encoder", self.cell_name, i),
                                                   kernel_initializer="glorot_uniform",
                                                   recurrent_initializer="glorot_uniform",
                                                   bias_initializer='zeros',
                                                   return_sequences=True),
                                     merge_mode="concat", dtype="float32")(q_encode)
        # q_encoder output shape: (None, hidden_size * 2)
        # TODO: 用最后一步的隐层状态表示q
        q_encode = Bidirectional(self.rnn_cell(units=hidden_size,
                                               name="{}-{}-{}".format("q-encoder", self.cell_name, num_layers),
                                               kernel_initializer="glorot_uniform",
                                               recurrent_initializer="glorot_uniform",
                                               bias_initializer='zeros',
                                               return_sequences=False),
                                 merge_mode="concat", dtype="float32")(q_encode)

        # 上下文文档的编码模型
        # output shape: (None, max_d_length, embedding_dim)
        d_encode = Embedding(input_dim=self.vocab_size,
                             output_dim=embedding_dim,
                             weights=[embedding_matrix],
                             mask_zero=True,
                             input_length=self.d_len)(d_input)

        # d_encoder output shape: (None, max_d_length, hidden_size * 2)
        for i in range(1, num_layers + 1):
            d_encode = Bidirectional(self.rnn_cell(units=hidden_size,
                                                   name="{}-{}-{}".format("d-encoder", self.cell_name, i),
                                                   kernel_initializer="glorot_uniform",
                                                   recurrent_initializer="glorot_uniform",
                                                   bias_initializer='zeros',
                                                   return_sequences=True),
                                     merge_mode="concat", dtype="float32")(d_encode)

        # noinspection PyUnusedLocal
        def my_dot(x):
            """注意力点乘函数，原始版本"""
            c = [tf.reduce_sum(tf.multiply(x[0][:, inx, :], x[1]), -1, keep_dims=True) for inx in range(self.d_len)]
            return tf.concat(c, -1)

        def my_dot_v2(x):
            """注意力点乘函数，快速版本"""
            d_btf, q_bf = x
            res = K.batch_dot(tf.expand_dims(q_bf, -1), d_btf, (1, 2))
            return K.reshape(res, [-1, self.d_len])

        mem_attention_pre_soft_bt = Lambda(my_dot_v2, name="attention")([d_encode, q_encode])
        mem_attention_pre_soft_masked_bt = Multiply(name="mask")([mem_attention_pre_soft_bt, context_mask])
        mem_attention_bt = Activation(activation="softmax", name="softmax")(mem_attention_pre_soft_masked_bt)

        # 注意力求和，attention-sum过程
        # TODO: Get rid of sentence-by-sentence processing?
        # TODO: Rewrite into matrix notation instead of scans?
        def sum_prob_of_word(word_ix, sentence_ixs, sentence_attention_probs):
            word_ixs_in_sentence = tf.where(tf.equal(sentence_ixs, word_ix))
            return tf.reduce_sum(tf.gather(sentence_attention_probs, word_ixs_in_sentence))

        # noinspection PyUnusedLocal
        def sum_probs_single_sentence(prev, cur):
            candidate_indices_i, sentence_ixs_t, sentence_attention_probs_t = cur
            result = tf.scan(
                fn=lambda previous, x: sum_prob_of_word(x, sentence_ixs_t, sentence_attention_probs_t),
                elems=[candidate_indices_i],
                initializer=tf.constant(0., dtype="float32"))
            return result

        def sum_probs_batch(candidate_indices_bi, sentence_ixs_bt, sentence_attention_probs_bt):
            result = tf.scan(
                fn=sum_probs_single_sentence,
                elems=[candidate_indices_bi, sentence_ixs_bt, sentence_attention_probs_bt],
                initializer=tf.Variable([0] * self.A_len, dtype="float32"))
            return result

        # output shape: (None, i) i = max_candidate_length = 10
        y_hat = Lambda(lambda x: sum_probs_batch(x[0], x[1], x[2]), name="attention_sum")(
            [candidates_bi, d_input, mem_attention_bt])
        self.model = Model(inputs=[q_input, d_input, context_mask, candidates_bi], outputs=y_hat)
        plot_model(self.model, to_file=__file__ + ".png", show_shapes=True, show_layer_names=True)
        self.model.summary()

    # noinspection PyUnusedLocal
    def train(self, train_data, valid_data, batch_size, epochs, opt_name, lr, grad_clip):
        """
        模型训练。
        """

        def save_weight_on_epoch_end(epoch, e_logs):
            filename = "{}weight-epoch{}-{}-{}.h5".format(self.weight_path,
                                                          time.strftime("%Y-%m-%d-(%H-%M)", time.localtime()),
                                                          epoch,
                                                          e_logs['val_acc'])
            self.model.save_weights(filepath=filename)

        checkpointer = LambdaCallback(on_epoch_end=save_weight_on_epoch_end)

        # tensorboard = TensorBoard(log_dir="./logs", histogram_freq=1, write_images=True)
        earlystopping = EarlyStopping(monitor="val_loss", patience=3, verbose=1)

        # 对输入进行预处理
        questions_ok, documents_ok, context_mask, candidates_ok, y_true = self.preprocess_input_sequences(train_data)
        v_questions, v_documents, v_context_mask, v_candidates, v_y_true = self.preprocess_input_sequences(valid_data)
        if opt_name == "SGD":
            optimizer = SGD(lr=lr, decay=1e-6, clipvalue=grad_clip)
        elif opt_name == "ADAM":
            optimizer = Adam(lr=lr, clipvalue=grad_clip)
        else:
            raise NotImplementedError("Other Optimizer Not Implemented.-_-||")
        self.model.compile(optimizer=optimizer,
                           loss="categorical_crossentropy",
                           metrics=["accuracy"])

        # 载入之前训练的权重
        self.load_weight()

        data = {"q_input": questions_ok,
                "d_input": documents_ok,
                "context_mask": context_mask,
                "candidates_bi": candidates_ok}
        v_data = {"q_input": v_questions,
                  "d_input": v_documents,
                  "context_mask": v_context_mask,
                  "candidates_bi": v_candidates}
        logs = self.model.fit(x=data,
                              y=y_true,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(v_data, v_y_true),
                              callbacks=[checkpointer, earlystopping])

    def test(self, test_data, batch_size):
        # 对输入进行预处理
        questions_ok, documents_ok, context_mask, candidates_ok, y_true = self.preprocess_input_sequences(test_data)
        data = {"q_input": questions_ok,
                "d_input": documents_ok,
                "context_mask": context_mask,
                "candidates_bi": candidates_ok}

        y_pred = self.model.predict(x=data, batch_size=batch_size)
        acc_num = np.count_nonzero(np.equal(np.argmax(y_pred, axis=-1), np.zeros(len(y_pred))))
        test_acc = acc_num / len(y_pred)
        logging.info("Test accuracy is {}".format(test_acc))
        return acc_num, test_acc

    def load_weight(self, weight_path=None):
        weight_file = self.weight_path if not weight_path else weight_path
        if os.path.exists(weight_file + "weight.h5"):
            logging.info("Load pre-trained weights:{}".format(weight_file + "weight.h5"))
            self.model.load_weights(filepath=weight_file + "weight.h5", by_name=True)

    @staticmethod
    def union_shuffle(data):
        d, q, a, A = data
        c = list(zip(d, q, a, A))
        random.shuffle(c)
        return zip(*c)

    def preprocess_input_sequences(self, data, shuffle=True):
        """
        预处理输入：
        shuffle
        PAD/TRUNC到固定长度的序列
        y_true是长度为self.A_len的向量，index=0为正确答案，one-hot编码
        """
        documents, questions, answer, candidates = self.union_shuffle(data) if shuffle else data
        d_lens = [len(i) for i in documents]

        questions_ok = pad_sequences(questions, maxlen=self.q_len, dtype="int32", padding="post", truncating="post")
        documents_ok = pad_sequences(documents, maxlen=self.d_len, dtype="int32", padding="post", truncating="post")
        context_mask = K.eval(tf.sequence_mask(d_lens, self.d_len, dtype=tf.float32))
        candidates_ok = pad_sequences(candidates, maxlen=self.A_len, dtype="int32", padding="post", truncating="post")
        y_true = np.zeros_like(candidates_ok)
        y_true[:, 0] = 1
        return questions_ok, documents_ok, context_mask, candidates_ok, y_true
