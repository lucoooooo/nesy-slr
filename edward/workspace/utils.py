import os
from abc import ABC, abstractmethod
import random
from typing import *
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3
import tensorflow_probability as tfp
import numpy as np
from tqdm import tqdm
from edward2 import tensorflow as ed
import time

tfd = tfp.distributions

def time_delta_now(t_start: float, simple_format=True) -> str:
    a = t_start
    b = time.perf_counter() 
    c = b - a  
    days = int(c // 86400)
    hours = int(c // 3600 % 24)
    minutes = int(c // 60 % 60)
    seconds = int(c % 60)
    millisecs = round(c % 1 * 1000)
    if simple_format:
        return f"{hours}h:{minutes}m:{seconds}s"

    return f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds, {millisecs} milliseconds", c

class MNISTSum2Dataset:
    def __init__(self, data_dir, train=True, seed:int = None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")
        
        if train:
            self.x, self.y = x_train, y_train
        else:
            self.x, self.y = x_test, y_test

        self.x = np.expand_dims(self.x, axis=-1).astype("float32")
        self.y = self.y.astype("int32")
        self.indices = list(range(len(self.x)))
        if seed is None:
            random.shuffle(self.indices)
        else:
            rng = random.Random(seed)
            rng.shuffle(self.indices)

    def __len__(self):
        return int(len(self.x) / 2)

    def generator(self):
        limit = len(self)
        for idx in range(limit):
            idx1 = self.indices[idx * 2]
            idx2 = self.indices[idx * 2 + 1]
            
            img_a = self.x[idx1]
            digit_a = self.y[idx1]
            
            img_b = self.x[idx2]
            digit_b = self.y[idx2]
            yield ((img_a, img_b), (digit_a, digit_b))

def mnist_sum_2_loader(data_dir, batch_size_train, batch_size_test, modeltype, train_dataset, test_dataset):
    def create_dataset_from_object(dataset_obj):

        indices = np.array(dataset_obj.indices)

        if len(indices) % 2 != 0:
            indices = indices[:-1]

        idx_a = indices[0::2]
        idx_b = indices[1::2]

        imgs_a = dataset_obj.x[idx_a]
        imgs_b = dataset_obj.x[idx_b]
        lbls_a = dataset_obj.y[idx_a]
        lbls_b = dataset_obj.y[idx_b]

        ds = tf.data.Dataset.from_tensor_slices(((imgs_a, imgs_b), (lbls_a, lbls_b)))
        return ds

    train_ds = create_dataset_from_object(train_dataset)
    test_ds = create_dataset_from_object(test_dataset)

    def process_batch(imgs, lbls):
        img1, img2 = imgs
        
        img1 = tf.cast(img1, tf.float32) / 255.0
        img2 = tf.cast(img2, tf.float32) / 255.0

        if "b0" in modeltype or "b3" in modeltype:
            img1 = tf.image.grayscale_to_rgb(img1)
            img2 = tf.image.grayscale_to_rgb(img2)
            img1 = tf.image.resize(img1, [64, 64])
            img2 = tf.image.resize(img2, [64, 64])
            
            mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
            std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
            img1 = (img1 - mean) / std
            img2 = (img2 - mean) / std
        else:
            img1 = (img1 - 0.1307) / 0.3081
            img2 = (img2 - 0.1307) / 0.3081
            
        return (img1, img2), lbls

    train_ds = (train_ds
                .batch(batch_size_train)
                .map(process_batch, num_parallel_calls=tf.data.AUTOTUNE)
                .cache() 
                .prefetch(tf.data.AUTOTUNE))

    test_ds = (test_ds
            .batch(batch_size_test)
            .map(process_batch, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .prefetch(tf.data.AUTOTUNE))

    return train_ds, test_ds

class MNISTNet_basic(keras.Model):
    def __init__(self, num_classes=10):
        super(MNISTNet_basic, self).__init__()
        self.modelname = "basic"
        self.conv1 = layers.Conv2D(6, kernel_size=5, activation='relu', kernel_initializer='he_uniform') #init pesi come pytorch
        self.pool = layers.MaxPooling2D(pool_size=2)
        self.conv2 = layers.Conv2D(16, kernel_size=5, activation='relu', kernel_initializer='he_uniform')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(120, activation='relu',kernel_initializer='he_uniform')
        self.fc2 = layers.Dense(84, activation='relu',kernel_initializer='he_uniform')
        self.fc3 = layers.Dense(num_classes,kernel_initializer='he_uniform')
        self.dropout = layers.Dropout(0.5)

    def call(self, x):
        x = self.conv1(x)       
        x = self.pool(x)
        x = self.conv2(x)    
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class MNISTNet_Efficient(keras.Model):
    def __init__(self, model_type="b0", N=10):
        super(MNISTNet_Efficient, self).__init__()
        self.modelname = model_type
        if model_type == "b0":
            base_model = EfficientNetB0(include_top=False, weights='imagenet')
        else:
            base_model = EfficientNetB3(include_top=False, weights='imagenet')
        
        base_model.trainable = True 
        
        self.feature_extractor = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D()
        ])
        self.classifier = layers.Dense(N, kernel_initializer='he_uniform')

    def call(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

class MNISTSum2Net_nesy(keras.Model):
    def __init__(self, base_model):
        super(MNISTSum2Net_nesy, self).__init__()
        self.mnist_net = base_model

    def call(self, inputs, training = None):
        (a_imgs, b_imgs) = inputs

        a_feat = self.mnist_net(a_imgs, training=training)
        b_feat = self.mnist_net(b_imgs, training=training)
        
        d1 = ed.RandomVariable(tfd.Categorical(logits=a_feat, name="digit_1"))
        d2 = ed.RandomVariable(tfd.Categorical(logits=b_feat, name="digit_2"))

        return d1,d2
    
    def _conv_sum(self, d1probs, d2probs):

        x = tf.transpose(d1probs, [1,0])
        x = tf.reshape(x,[1,1,10,-1])

        y = tf.reverse(d2probs, axis=[1]) #flippo il vettore per poi fare P(S = k) = sum_i(P(d1 = i) * P(d2=k-i)))
        y = tf.transpose(y, [1, 0]) 
        y = tf.reshape(y, [1, 10, -1, 1])
        
        padding = [[0, 0], [0, 0], [9, 9], [0, 0]]
        X_pad = tf.pad(x, padding)
        
        out = tf.nn.depthwise_conv2d(
            X_pad, 
            y, 
            strides=[1, 1, 1, 1], 
            padding='VALID'
        )
        sum_prob = tf.reshape(out, [19, -1])
        sum_prob = tf.transpose(sum_prob, [1, 0])
    
        sum_prob = sum_prob / (tf.reduce_sum(sum_prob, axis=1, keepdims=True) + 1e-9)
    
        return sum_prob
    
    #usando variabili discreti serve ottenere una loss differenziabile per fare backprop=> come per pytorch uso la somma convoluttiva
    def diff_loss(self, y_true, d1, d2, loss_fn):
        d1probs = d1.distribution.probs_parameter()
        d2probs = d2.distribution.probs_parameter()
        
        sum_prob = self._conv_sum(d1probs=d1probs, d2probs=d2probs)

        loss = loss_fn(y_true, sum_prob + 1e-9)
        return loss

class MNISTSum2Net(keras.Model):
    def __init__(self, base_model):
        super(MNISTSum2Net, self).__init__()
        self.mnist_net = base_model
        self.sum_classifier = layers.Dense(19, kernel_initializer='he_uniform')

    def call(self, inputs, training=None):
        (a_imgs, b_imgs) = inputs

        a_feat = self.mnist_net(a_imgs, training)
        b_feat = self.mnist_net(b_imgs, training)
        
        combined_feature = layers.concatenate([a_feat, b_feat], axis=1)
        sum_logits = self.sum_classifier(combined_feature, training)
        
        sum_probs = tf.nn.softmax(sum_logits)
        
        a_pred = tf.argmax(tf.nn.softmax(a_feat), axis=1)
        b_pred = tf.argmax(tf.nn.softmax(b_feat), axis=1)

        return sum_probs, a_pred, b_pred

class Trainer_NoSym:
    def __init__(self, train_loader, test_loader, model_dir, learning_rate, model: MNISTSum2Net):
        self.model_dir = model_dir
        self.network = model
        self.optimizer = optimizers.Adam(learning_rate=learning_rate,epsilon=1e-8)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = losses.SparseCategoricalCrossentropy(from_logits=False) 

    @tf.function 
    def train_step(self, img1, img2, d1, d2):
        target_sum = d1 + d2
        with tf.GradientTape() as tape:
            output, d1_pred, d2_pred = self.network((img1, img2), training=True)
            loss_value = self.loss_fn(target_sum, output + 1e-9)

        grads = tape.gradient(loss_value, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_weights))
            
        return loss_value, output, d1_pred, d2_pred

    def train(self, num_epoch):
        train_losses, acc_train = [], []
        d1_acc_train, d2_acc_train = [], []

        for epoch in range(num_epoch):

            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.Accuracy()
            epoch_d1_acc = tf.keras.metrics.Accuracy()
            epoch_d2_acc = tf.keras.metrics.Accuracy()

            for ((img1, img2), (d1, d2)) in tqdm(self.train_loader, desc='Train Loop'):
                
                loss_value, output, d1_pred, d2_pred = self.train_step(img1, img2, d1, d2)
                epoch_loss_avg.update_state(loss_value)
                epoch_accuracy.update_state(d1+d2, tf.argmax(output, axis=1))
                epoch_d1_acc.update_state(d1, d1_pred)
                epoch_d2_acc.update_state(d2, d2_pred)

            tl = epoch_loss_avg.result().numpy()
            ta = epoch_accuracy.result().numpy()
            d1a = epoch_d1_acc.result().numpy()
            d2a = epoch_d2_acc.result().numpy()

            train_losses.append(float(tl))
            acc_train.append(float(ta))
            d1_acc_train.append(float(d1a))
            d2_acc_train.append(float(d2a))

            print(f"\nEpoca {epoch+1}/{num_epoch} - {self.network.mnist_net.modelname} neural model - Train loss: {tl} with total accuracy: {ta*100}% \n digit1 accuracy: {d1a*100} and digit2 accuracy: {d2a*100}\n")

        self.network.save_weights(os.path.join(self.model_dir, f"{self.network.mnist_net.modelname}_neural.weights.h5"))
        
        return {
            "loss": train_losses,
            "accuracy" : acc_train,
            "single-digit accuracy": (max(d1_acc_train), max(d2_acc_train))
        }

    def test(self):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.Accuracy()
        epoch_d1_acc = tf.keras.metrics.Accuracy()
        epoch_d2_acc = tf.keras.metrics.Accuracy()

        for ((img1, img2), (d1, d2)) in tqdm(self.test_loader, desc='Test Loop'):
            target_sum = d1 + d2
            
            output, d1_pred, d2_pred = self.network((img1, img2), training=False)
            
            loss_value = self.loss_fn(target_sum, output + 1e-9)
            
            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(target_sum, tf.argmax(output, axis=1))
            epoch_d1_acc.update_state(d1, d1_pred)
            epoch_d2_acc.update_state(d2, d2_pred)

        tl = float(epoch_loss_avg.result().numpy())
        ta = float(epoch_accuracy.result().numpy())
        d1a = float(epoch_d1_acc.result().numpy())
        d2a = float(epoch_d2_acc.result().numpy())

        print(f"\n- {self.network.mnist_net.modelname} neural model - Test loss: {tl} with total accuracy: {ta*100}% \n digit1 accuracy: {d1a*100} and digit2 accuracy: {d2a*100}\n")
        
        return {
            "loss": tl,
            "accuracy" : ta,
            "single-digit accuracy": (d1a, d2a)
        }

class Trainer_Sym:
    def __init__(self, train_loader, test_loader, model_dir, model: MNISTSum2Net_nesy, learning_rate):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_dir = model_dir
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, epsilon=1e-8) #metto epsilon come il default di pytorch
        self.network = model 
        self.loss_fn = losses.SparseCategoricalCrossentropy(from_logits=False) 

    @tf.function
    def train_step(self, img1, img2, target_sum):
        with tf.GradientTape() as tape:
            d1_rv, d2_rv = self.network((img1, img2), training=True)
            loss_value = self.network.diff_loss(target_sum, d1_rv, d2_rv, loss_fn=self.loss_fn)

        grads = tape.gradient(loss_value, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_weights))
        d1_logits = d1_rv.distribution.logits
        d2_logits = d2_rv.distribution.logits
        return loss_value, d1_logits, d2_logits

    def train(self, num_epoch):
        train_losses, acc_train = [], []
        d1_acc_train, d2_acc_train = [], []

        for epoch in range(num_epoch):
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.Accuracy() 
            epoch_d1_acc = tf.keras.metrics.Accuracy()  
            epoch_d2_acc = tf.keras.metrics.Accuracy()   

            for ((img1, img2), (d1_gt, d2_gt)) in tqdm(self.train_loader, desc='Train Loop'):
                target_sum = d1_gt + d2_gt

                loss_value, d1_rv_logits, d2_rv_logits = self.train_step(img1, img2, target_sum)
                epoch_loss_avg.update_state(loss_value)
                
                d1_pred = tf.argmax(d1_rv_logits, axis=1, output_type=tf.int32)
                d2_pred = tf.argmax(d2_rv_logits, axis=1, output_type=tf.int32)
                sum_pred = d1_pred + d2_pred
                
                epoch_accuracy.update_state(target_sum, sum_pred)
                epoch_d1_acc.update_state(d1_gt, d1_pred)
                epoch_d2_acc.update_state(d2_gt, d2_pred)

            tl = epoch_loss_avg.result().numpy()
            ta = epoch_accuracy.result().numpy()
            d1a = epoch_d1_acc.result().numpy()
            d2a = epoch_d2_acc.result().numpy()
            
            train_losses.append(float(tl))
            acc_train.append(float(ta))
            d1_acc_train.append(float(d1a))
            d2_acc_train.append(float(d2a))

            print(f"\nEpoca {epoch+1}/{num_epoch} - {self.network.mnist_net.modelname} nesy model - Train loss: {tl} with total accuracy: {ta*100}% \n digit1 accuracy: {d1a*100} and digit2 accuracy: {d2a*100}\n")

        self.network.save_weights(os.path.join(self.model_dir, f"{self.network.mnist_net.modelname}_nesy.weights.h5"))
        return {
            "loss": train_losses,
            "accuracy" : acc_train,
            "single-digit accuracy": (max(d1_acc_train), max(d2_acc_train))
        }

    def test(self):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.Accuracy()
        epoch_d1_acc = tf.keras.metrics.Accuracy()
        epoch_d2_acc = tf.keras.metrics.Accuracy()

        for ((img1, img2), (d1_gt, d2_gt)) in tqdm(self.test_loader, desc='Test Loop'):
            target_sum = d1_gt + d2_gt
            
            d1_rv, d2_rv = self.network((img1, img2), training=False)
            loss_value = self.network.diff_loss(target_sum, d1_rv, d2_rv, self.loss_fn)
            
            d1_pred = tf.argmax(d1_rv.distribution.logits, axis=1, output_type=tf.int32)
            d2_pred = tf.argmax(d2_rv.distribution.logits, axis=1, output_type=tf.int32)
            sum_pred = d1_pred + d2_pred

            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(target_sum, sum_pred)
            epoch_d1_acc.update_state(d1_gt, d1_pred)
            epoch_d2_acc.update_state(d2_gt, d2_pred)

        tl = float(epoch_loss_avg.result().numpy())
        ta = float(epoch_accuracy.result().numpy())
        d1a = float(epoch_d1_acc.result().numpy())
        d2a = float(epoch_d2_acc.result().numpy())

        print(f"\n- {self.network.mnist_net.modelname} nesy model - Test loss: {tl} with total accuracy: {ta*100}% \n digit1 accuracy: {d1a*100} and digit2 accuracy: {d2a*100}\n")
        return {
            "loss": tl,
            "accuracy" : ta,
            "single-digit accuracy": (d1a, d2a)
        }