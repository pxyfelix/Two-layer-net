#作业要求 
#1.训练：
# 激活函数
# 反向传播，loss以及梯度的计算
# 学习率下降策略
# L2正则化
# 优化器SGD
# 保存模型

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 读取mnist数据集，方法参考自https://blog.csdn.net/simple_the_best/article/details/75267863
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

# 使用numpy构造两层神经网络，参考自https://zhuanlan.zhihu.com/p/25268643
# import numpy as np

# 定义ReLU函数


def ReLU(x):
    return np.maximum(0, x)


class Two_Layer_Net():
    """
    一个二层的全连接网络 输入层的维度为D, 隐藏层维度为 H, 类别为C类.

    input -> fully connected layer -> ReLU -> fully connected layer -> softmax
    第二个全连接层的输出为分数
    """

    def __init__(self, input_size=0, hidden_size=0, output_size=0, std=1e-4):
        """

        W1:  (D, H)
        b1:  (H,)
        W2:  (H, C)
        b2:  (C,)
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros((1,hidden_size))
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros((1,output_size))

    # 定义损失函数

    def loss(self, X, y, regularization=0.0):

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        hidden1 = ReLU(X.dot(W1)+b1)      # 隐藏层1  (N,H)
        output = hidden1.dot(W2)+b2       # 输出层  (N,C)

        softmax = np.exp(output)  # Softmax (N,C)
        for i in range(0, N):
            softmax[i, :] /= np.sum(softmax[i, :])

        loss = 0  # loss
        data_loss=0
        for i in range(0, N):
            data_loss += -np.log(softmax[i, y[i]])
        reg_loss = 0.5*regularization*(np.sum(W1*W1)+np.sum(W2*W2))
        loss = data_loss/N + reg_loss

        # 反向传播
        gradient = {}
        dl = softmax.copy()
        for i in range(0, N):
            dl[i, y[i]] -= 1
        dl /= N
        gradient['W2'] = hidden1.T.dot(dl) + regularization*W2  # (H,C)
        gradient['b2'] = np.sum(dl, axis=0, keepdims=True)  # (1,C)
        # ReLU层
        dh1 = dl.dot(W2.T)
        dh1 = (hidden1 > 0)*dh1    # (N,H)
        gradient['W1'] = X.T.dot(dh1) + regularization*W1  # (D,H)
        gradient['b1'] = np.sum(dh1, axis=0, keepdims=True)  # (1,H)

        return loss, gradient

    def train(self, X_train, y_train, X_test, y_test,
              learning_rate=5e-3, lr_decay=0.9,decay_steps = 100,
              regulariaztion=1e-3, iteration=300, batch_size=1000,mu=0.9,mu_increase=1.0,):
    
        """    
        使用优化器SGD 
        Inputs:    
        - X_train:  (N, D)    
        - y_train:(N,) 
        - X_test:  (N_test, D)     
        - y_test:  (N_test,)     
        - learning_rate: 学习率    
        - lr_decay: 学习率衰减因子
        - reg: L2正则化  
        - batch_size:     
        """

        N = X_train.shape[0]
        v_W2, v_b2 = 0.0, 0.0
        v_W1, v_b1 = 0.0, 0.0
        train_loss_history = []
        test_loss_history = []
        test_accuracy_history = []            
        X_batch = None
        y_batch = None

        
        for i in tqdm(range(iteration)):


            sample_index = np.random.choice(N,batch_size,replace =True)
            X_batch = X_train[sample_index]
            y_batch = y_train[sample_index]

            loss, grads = self.loss(X_batch, y_batch, regulariaztion)
            train_loss_history.append(loss)


            # SGD结合向量
            v_W2 = mu * v_W2 - learning_rate * grads['W2']
            self.params['W2'] += v_W2
            v_b2 = mu * v_b2 - learning_rate * grads['b2']
            self.params['b2'] += v_b2
            v_W1 = mu * v_W1 - learning_rate * grads['W1']
            self.params['W1'] += v_W1
            v_b1 = mu * v_b1 - learning_rate * grads['b1']
            self.params['b1'] += v_b1

            loss, grads = self.loss(X_test, y_test, regulariaztion)
            test_loss_history.append(loss)
            test_accuracy = (self.predict(X_test) == y_test).mean()
            test_accuracy_history.append(test_accuracy)
            if i % decay_steps ==0:
                learning_rate = learning_rate * lr_decay
                # mu也要变化
                mu *= mu_increase
        return train_loss_history,test_loss_history,test_accuracy_history
    
    def predict(self, X):
        y_pred = None
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        hidden1 = ReLU(X.dot(W1)+b1)
        output = hidden1.dot(W2)+b2
        y_pred = np.argmax(output, axis=1)

        return y_pred

    #使用numpy中savez函数保存模型
    def save_model(self, file):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        np.savez(file, W1=W1, b1=b1, W2=W2, b2=b2)

    def load_model(self, file):
        data = np.load(file)
        self.params['W1'] = data['W1']
        self.params['b1'] = data['b1']
        self.params['W2'] = data['W2']
        self.params['b2'] = data['b2']

if __name__ == "__main__":

    np.random.seed(123)
    train_images, train_labels = load_mnist('./mnist')
    test_images, test_labels = load_mnist('./mnist', 't10k')
    print('Train data shape: ', train_images.shape)
    print('Train labels shape: ', train_labels.shape)
    print('Test data shape: ', test_images.shape)
    print('Test labels shape: ', test_labels.shape)

    twolayermodel = Two_Layer_Net(train_images.shape[1], 100, 10)
    train_loss_history, test_loss_history, test_acc_history = twolayermodel.train(train_images, train_labels, test_images, test_labels)

    print(test_acc_history)

    twolayermodel.save_model('./twolayermodel.npz')
    #可视化训练和测试的loss曲线，测试的accuracy曲线
    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.plot(train_loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.title('Training Loss history')

    plt.figure(1)
    plt.subplot(3, 1, 2)
    plt.plot(test_loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Testing Loss')
    plt.title('Testing Loss history')

    plt.figure(1)
    plt.subplot(3, 1, 3)
    plt.plot(test_acc_history)
    plt.xlabel('Iteration')
    plt.ylabel('Testing Accuracy')
    plt.title('Testing Accuracy history')

    plt.show()


