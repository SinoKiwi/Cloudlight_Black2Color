'''
用plaidml作为keras的后端实现一个UNET，用于将黑白图片转换为彩色图片并以原分辨率输出，即输入为黑白图片，输出为彩色图片
训练集在./data目录下，均为512*512彩色图片，数据预处理时先用PIL处理为黑白图片再送入模型训练预测
'''
import plaidml.keras
plaidml.keras.install_backend()
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
from PIL import Image
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
import argparse


# 定义UNET模型
def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    # 用keras的functional api构建模型，输入为512*512的黑白图片，输出为512*512的彩色图片
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(3, 3, activation='linear', padding='same', kernel_initializer='he_normal')(conv9)
    model = Model(input=inputs, output=conv9)

    # 加载预训练权重
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model


# 数据预处理，将数据集读入numpy并转换为黑白图片
def data_pre(data_path, type):
    fileNames = os.listdir(data_path)
    data_num = len(fileNames)
    data = np.ndarray((data_num, 256, 256, 1), dtype=np.uint8)
    label = np.ndarray((data_num, 256, 256, 3), dtype=np.uint8)
    for i, fileName in enumerate(fileNames):
        #读入彩色图像，作为label
        img = Image.open(os.path.join(data_path, fileName))
        npimg = np.array(img)
        label[i] = npimg
        #转换为黑白图像，作为data
        img = img.convert('L')
        img = np.array(img)
        img = img[:, :, np.newaxis]
        img = img.reshape(1, 256, 256, 1)
        data[i] = img
    print(data[1].shape)
    print(label[1].shape)
    return data, label

# 训练模型
def train(train_data, train_label):
    # 训练模型
    model = unet()
    model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])
    model_checkpoint = ModelCheckpoint('unet_plaidml.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model.fit(train_data, train_label, batch_size=1, epochs=50, 
              callbacks=[model_checkpoint])


# 模型预测
def predict(image_path, predict_path):
    # 读入图片并转换为黑白图片
    img = np.array(Image.open(os.path.join(data_path, fileName)).convert('L'))
    img = img[:, :, np.newaxis]
    img = img.reshape(1, 256, 256, 1)
    # 加载模型并预测
    model = unet()
    model.load_weights('unet_plaidml.hdf5')
    predict = model.predict(img, batch_size=1, verbose=1)
    # 将预测结果转换为彩色图片并保存
    predict = predict.reshape((256, 256, 3))
    predict = predict * 255
    predict = predict.astype(np.uint8)
    p = Image.fromarray(predict)
    p.save(predict_path)


if __name__ == '__main__':
    # 设定程序的运行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path', default='./data', type=str)
    parser.add_argument('--image_path', dest='image_path', default='./predict_image/predict1.jpg', type=str)
    parser.add_argument('--predict_path', dest='predict_path', default='./predict_image/predict1_predict.jpg', type=str)
    parser.add_argument('-m', dest='m', default='train', type=str)
    args = parser.parse_args()

    if args.m == 'train':
        # 训练模型
        train_data, train_label = data_pre(args.data_path, 'train')
        train(train_data, train_label)

    else:
        # 模型预测
        predict(args.image_path, args.predict_path)