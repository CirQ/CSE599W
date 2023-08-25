import os
import urllib
import gzip
import struct

import numpy as np
from mxnet import nd, gluon, autograd, metric

import autodiff as ad


def read_data(label_url, image_url):
    with gzip.open(label_url) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.frombuffer(flbl.read(), dtype=np.int8)
    with gzip.open(image_url, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.frombuffer(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)

def to4d(img):
    return img.reshape(img.shape[0], 784).astype(np.float32)/256

def to1hot(lbl):
    return np.eye(10)[lbl]

def data_iter(batch_size):
    (train_lbl, train_img) = read_data('../train-labels-idx1-ubyte.gz', '../train-images-idx3-ubyte.gz')
    (test_lbl, test_img) = read_data('../t10k-labels-idx1-ubyte.gz', '../t10k-images-idx3-ubyte.gz')
    train_load = gluon.data.DataLoader(gluon.data.ArrayDataset(to4d(train_img), to1hot(train_lbl)), batch_size, shuffle=True)
    test_load = gluon.data.DataLoader(gluon.data.ArrayDataset(to4d(test_img), to1hot(test_lbl)), batch_size)
    return train_load, test_load

def evaluate_accuracy(data_iterator, trainer, x, W, yh, W_val):
    acc = metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        x_val = data.asnumpy()
        yh_val = label.asnumpy()
        output, _, _ = trainer.run(feed_dict={x:x_val, W:W_val, yh:yh_val})
        preds = nd.array(output.argmax(axis=1))
        labels = nd.array(label.argmax(axis=1))
        acc.update(preds=preds, labels=labels)
    return acc.get()[1]



def main():
    train_iter, test_iter = data_iter(batch_size=16)

    learning_rate = 0.005

    x = ad.Variable(name='x')
    W = ad.Variable(name='W')
    logits = ad.matmul_op(x, W)
    y = 1 / (1 + ad.exp_func(-logits))

    yh = ad.Variable(name='yh')
    loss = - ad.reduce_sum_op(yh * ad.log_func(y))

    grad_W, = ad.gradients(loss, [W])

    trainer = ad.Executor([y, loss, grad_W])

    W_val = np.random.normal(size=(784, 10))
    for e in range(10):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_iter):
            x_val = data.asnumpy()
            yh_val = label.asnumpy()
    
            _, loss_val, grad_W_val = trainer.run(feed_dict={x:x_val, W:W_val, yh:yh_val})

            W_val = W_val - learning_rate * grad_W_val
            cumulative_loss += loss_val[0][0]

        train_accuracy = evaluate_accuracy(train_iter, trainer, x, W, yh, W_val)
        test_accuracy = evaluate_accuracy(test_iter, trainer, x, W, yh, W_val)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e+1, cumulative_loss, train_accuracy, test_accuracy))

    img, lbl = list(test_iter)[0]
    print(img[0])
    print(lbl[0])
    output = net(img)
    print("Predicted digit is", nd.argmax(output, axis=1).asnumpy().astype(np.int8)[0])



main()
