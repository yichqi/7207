# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as scio
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split 


from keras.models import Sequential 
from keras.layers.core import Dense
from keras.layers import Activation
from keras.optimizers import RMSprop
from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomUniform, Initializer, Constant

np.random.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import matplotlib.pyplot as plt



class InitCentersKMeans(Initializer):
    """ Initializer for initialization of centers of RBF network
        by clustering the given data set.
    # Arguments
        X: matrix, dataset
    """

    def __init__(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]

        n_centers = shape[0]
        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)
        return km.cluster_centers_



class RBFLayer(Layer):
    """ Layer of Gaussian RBF units.
    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the
                    layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas
    """

    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=Constant(
                                         value=self.init_betas),
                                     # initializer='ones',
                                     trainable=True)

        super(RBFLayer, self).build(input_shape)

    def call(self, x):

        C = K.expand_dims(self.centers)
        H = K.transpose(C-K.transpose(x))
        return K.exp(-self.betas * K.sum(H**2, axis=1))

        # C = self.centers[np.newaxis, :, :]
        # X = x[:, np.newaxis, :]

        # diffnorm = K.sum((C-X)**2, axis=-1)
        # ret = K.exp( - self.betas * diffnorm)
        # return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == '__main__':
    
    # 读取数据 
    xtrain = scio.loadmat('./data/data_train.mat')['data_train']
    ytrain = scio.loadmat('./data/label_train.mat')['label_train']
    xtest = scio.loadmat('./data/data_test.mat')['data_test']
    
    # 数据归一化
    scaler = MinMaxScaler()
    xtrain =scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    
    # 划分训练集和验证集
    xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.2, random_state=0)
    
    # 标签杜热编码
    ohe = OneHotEncoder()
    ytrain = ohe.fit_transform(ytrain).toarray()
    yval = ohe.transform(yval).toarray()
    
    # 对k进行参数寻优
    acc = -1
    accuracies = []
    BestModel = None
    BestHistory = None 
    for k in range(10, 200,10):
        print(k)
        model = Sequential()
        rbflayer = RBFLayer(k,
                            initializer=InitCentersKMeans(xtrain, 100),
                            betas=1,
                            input_shape=(33,))
        model.add(rbflayer)
        model.add(Dense(2, kernel_initializer='zeros', bias_initializer='zeros'))
        model.add(Activation('linear'))
        model.compile(loss='mean_squared_error',
                          optimizer=RMSprop(), metrics=['accuracy'])
        # print(model.summary())
        history1 = model.fit(xtrain, ytrain, 
                             epochs=300, 
                             batch_size=128, 
                             validation_data=(xval, yval),
                             verbose=False)
        pred_train = model.predict(xtrain)
        pred_val = model.predict(xval)
        trn_acc = sum(np.argmax(pred_train, axis=1)==np.argmax(ytrain, axis=1))/ytrain.shape[0]
        val_acc = sum(np.argmax(pred_val, axis=1)==np.argmax(yval, axis=1))/yval.shape[0]
        accuracies.append((k, trn_acc, val_acc))
        
        if acc < val_acc:
            acc = val_acc 
            BestModel = model 
            BestHistory = history1 


    # 最优模型
    model = BestModel
    history1 = BestHistory
    print(model.summary())
    
    # 参数寻优曲线
    accuracies = np.array(accuracies)
    plt.figure()
    plt.plot(accuracies[:, 0], accuracies[:, 1], label='train')
    plt.plot(accuracies[:, 0], accuracies[:, 2], label='val')
    plt.xlabel('K')
    plt.ylabel('accuracy')
    plt.title('accuracy of different K')
    plt.legend()
    
    # 最优模型训练过程曲线
    plt.figure()
    plt.plot(history1.history['acc'])
    plt.plot(history1.history['loss'])
    plt.plot(history1.history['val_acc'])
    plt.plot(history1.history['val_loss'])
    plt.title('train and val accuracy and loss')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['accuracy', 'loss', 'val accuracy', 'val loss'], loc='upper left')
    plt.show()
    
    
    # 训练集结果输出
    pred_trn = model.predict(xtrain)
    
    # 训练集结果评价
    con_mat = confusion_matrix(np.argmax(ytrain, axis=1), np.argmax(pred_trn, axis=1))
    print('训练集混淆矩阵：\n', con_mat)
    tp, fp, tn, fn = con_mat[0,0], con_mat[1,0], con_mat[1,1], con_mat[0,1]
    tpr0 = tp / (tp + fn)
    fpr0 = fp / (fp + tn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1score = 2 * tp / (2*tp + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    print('tpr:{}\nfpr:{}\naccuracy:{}\nprecision:{}\nrecall:{}\nf1score:{}\nsensitivity:{}\nspecificity:{}\n'.format(tpr0, fpr0, accuracy, precision, recall, f1score, sensitivity, specificity))
    
    precision, recall, thresholds = precision_recall_curve(np.argmax(ytrain, axis=1), np.max(pred_trn, axis=1))
    plt.figure("P-R Curve")
    plt.title('Precision/Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(recall,precision)
    plt.show()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(ytrain[:, i], pred_trn[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure()
    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange',
              lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    
    # 验证集结果输出
    pred_val = model.predict(xval)
    
    # 验证集结果评价
    con_mat = confusion_matrix(np.argmax(yval, axis=1), np.argmax(pred_val, axis=1))
    print('训练集混淆矩阵：\n', con_mat)
    tp, fp, tn, fn = con_mat[0,0], con_mat[1,0], con_mat[1,1], con_mat[0,1]
    tpr0 = tp / (tp + fn)
    fpr0 = fp / (fp + tn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1score = 2 * tp / (2*tp + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    print('tpr:{}\nfpr:{}\naccuracy:{}\nprecision:{}\nrecall:{}\nf1score:{}\nsensitivity:{}\nspecificity:{}\n'.format(tpr0, fpr0, accuracy, precision, recall, f1score, sensitivity, specificity))
    
    precision, recall, thresholds = precision_recall_curve(np.argmax(yval, axis=1), np.max(pred_val, axis=1))
    plt.figure("P-R Curve")
    plt.title('Precision/Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(recall,precision)
    plt.show()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(yval[:, i], pred_val[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure()
    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange',
              lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # 测试集结果输出
    pred_test = model.predict(xtest)
    print(pred_test)
    print(pred_trn)

        
