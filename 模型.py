#
import keras
import numpy as np
from keras import regularizers
from keras.layers import Input, Dense, Reshape, Flatten, Dropout,Embedding,LSTM,Concatenate
from keras.layers import BatchNormalization, Activation
from keras.layers import LeakyReLU     
from keras.layers import UpSampling2D, Conv2D     
from keras.models import Model
from keras.optimizers import Adam
from keras import models
from PIL import Image
import matplotlib.pyplot as plt
from random import randint
import 加载
word_num = 2500
describe_long=540
class SharedEmbedder:
    def __init__(self):
        self.model = None
        self._build_embedder()
    def _build_embedder(self):
        try:
            self.model = models.load_model('模型/emb.keras')
            print('编码器加载成功')
        except:
            print('编码器加载失败')
            word = Input(shape=(describe_long,))
            word_1 = Embedding(word_num, 10)(word)
            word_2 = LSTM(64, activation='sigmoid',return_sequences=True)(word_1)
            word_3 = LSTM(256, activation='sigmoid')(word_2)
            word_4 = Reshape((16, 16, 1))(word_3)
            self.model = Model(word, word_4)
    def __call__(self, inputs):
        return self.model(inputs)
    def get_model(self):
        return self.model
    def save(self):
        self.model.save('模型/emb.keras')
# 创建全局编码器实例
shared_embedder = SharedEmbedder()
class Draw_model_64():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        #图片长，宽，高
        self.dis_losses=[]
        self.gen_losses=[]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)             
        self.latent_dim = 100 #噪声图
        optimizer1=Adam(0.0002,0.5)
        optimizer2=Adam(0.0002,0.5)
        self.generator = self.build_generator()
        self.discriminator=self.build_discriminator()
        try:
            self.discriminator=models.load_model('模型/disc64.keras')
            print('加载成功')
        except ValueError:
            print('加载失败')
        try:
            self.generator=models.load_model('模型/gene64.keras')
            print('加载成功')
        except ValueError:
            print('加载失败')
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer1, metrics=['accuracy'])
        z1=Input(shape=(describe_long,))
        z2 = Input(shape=(self.latent_dim,))
        img = self.generator([z1,z2])
        self.discriminator.trainable = False
        valid = self.discriminator([z1,img])
        self.combined = Model([z1,z2], valid)
        self.combined.compile(loss='mse', optimizer=optimizer2)
    def build_generator(self):
        #生成图片64x64
        word=Input(shape=(describe_long,))
        emb=shared_embedder(word)
        noise=Input(shape=(100,))
        x1=Dense(128 * 16 * 16, activation="relu", input_dim=100, kernel_regularizer=regularizers.L2(0.01),
                 kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(noise)
        x2=Reshape((16,16,128))(x1)
        com=Concatenate()([x2,emb])
        x3=UpSampling2D()(com)
        x4=Conv2D(64, kernel_size=3, padding="same",kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(x3)
        x5=BatchNormalization(momentum=0.8)(x4)
        x6=Activation("relu")(x5)
        x7=UpSampling2D()(x6)
        x8=Conv2D(32, kernel_size=3, padding="same", kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(x7)
        x9=BatchNormalization(momentum=0.8)(x8)
        x10=Activation("relu")(x9)
        x11=Conv2D(self.channels, kernel_size=3, padding="same",kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(x10)
        output=Activation("tanh")(x11)
        gen=Model([word,noise], output)
        gen.summary()
        return gen
    def build_discriminator(self):
        word=Input(shape=(None,))
        emb=shared_embedder(word)
        img = Input(shape=self.img_shape)
        x1=Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same")(img)
        x3=LeakyReLU(alpha=0.2)(x1)
        x4=Dropout(0.25)(x3)
        x5=Conv2D(64, kernel_size=3, strides=2, padding="same")(x4)
        x6=BatchNormalization(momentum=0.8)(x5)
        x7=LeakyReLU(alpha=0.2)(x6)
        x8=Dropout(0.25)(x7)
        com=Concatenate()([x8,emb])
        x9=Conv2D(128, kernel_size=3, strides=2, padding="same")(com)
        x10=BatchNormalization(momentum=0.8)(x9)
        x11=LeakyReLU(alpha=0.2)(x10)
        x12=Dropout(0.25)(x11)
        x13=Flatten()(x12)
        output=Dense(1, activation='sigmoid')(x13)
        validity = Model([word,img],output)
        validity.summary()
        return validity
    def train_on_batch(self,x_train,y_train,z_train,seed_=1):
        batch_size=x_train.shape[0]
        fake = np.zeros((batch_size, 1))
        vaule = np.ones((batch_size, 1))
        optimizer3 = Adam(0.0002, 0.5)
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        gen = self.generator.predict([z_train, noise])
        train_tag = np.concatenate((z_train,z_train), axis=0)
        train_img = np.concatenate((x_train, gen), axis=0)
        train_valid = np.concatenate((y_train, fake), axis=0)
        np.random.seed(seed_)
        np.random.shuffle(train_img)
        np.random.seed(seed_)
        np.random.shuffle(train_valid)
        np.random.seed(seed_)
        np.random.shuffle(train_tag)
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer3)
        d_loss = self.discriminator.train_on_batch([train_tag, train_img], train_valid)
        self.discriminator.trainable = False
        g_loss = self.combined.train_on_batch([z_train, noise], vaule)
        self.dis_losses.append(d_loss[0])
        self.gen_losses.append(g_loss)
    def save(self):
        self.discriminator.save('模型/disc64.keras')
        self.generator.save('模型/gene64.keras')
    def yuce(self,tag):
        if len(tag.shape)==1:
            tag=np.expand_dims(tag,0)
        noise = np.random.normal(0, 1, (tag.shape[0], self.latent_dim))  # 生成燥声数据
        gen_imgs = self.generator.predict([tag,noise])  # 生成图像
        gen_imag = (gen_imgs + 1) * 127.5
        gen_image = np.round(gen_imag[0], decimals=0)
        gen_image = np.clip(gen_image, 0, 255).astype(np.uint8)
        return gen_image,gen_imgs

class Draw_model_256():
    def __init__(self):
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        # 图片长，宽，高
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        optimizer1 = Adam(0.0002, 0.5)
        optimizer2 = Adam(0.0002, 0.5)
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        try:
            self.discriminator = models.load_model('模型/disc256.keras')
            print('加载成功')
        except ValueError:
            print('加载失败')
        try:
            self.generator = models.load_model('模型/gene256.keras')
            print('加载成功')
        except ValueError:
            print('加载失败')
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer1, metrics=['accuracy'])
        z1 = Input(shape=(describe_long,))
        z2 = Input(shape=(64,64,3))
        img = self.generator([z1, z2])
        self.discriminator.trainable = False
        valid = self.discriminator([z1, img])
        self.combined = Model([z1, z2], valid)
        self.combined.compile(loss='mse', optimizer=optimizer2)
    def build_generator(self):
        # 生成图片256x256
        word = Input(shape=(describe_long,))
        img64 = Input(shape=(64,64,3))
        img64_1=Conv2D(64, kernel_size=3, strides=2, padding="same",
                       kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(img64)
        img64_2=Activation('relu')(img64_1)
        img64_3=Conv2D(128, kernel_size=3, strides=2, padding="same",
                       kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(img64_2)
        img64_4=BatchNormalization(momentum=0.8)(img64_3)
        img64_5=Activation('relu')(img64_4)
        emb = shared_embedder(word)
        com = Concatenate()([emb,img64_5])
        x4 = Conv2D(512, kernel_size=3, padding="same",
                    kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(com)
        x5 = BatchNormalization(momentum=0.8)(x4)
        x6 = Activation("relu")(x5)
        x7 = UpSampling2D()(x6)
        x8 = Conv2D(256, kernel_size=3, padding="same",
                    kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(x7)
        x9 = BatchNormalization(momentum=0.8)(x8)
        x10 = Activation("relu")(x9)
        x11 = UpSampling2D()(x10)
        x12 = Conv2D(128, kernel_size=3, padding="same")(x11)
        x13 = BatchNormalization(momentum=0.8)(x12)
        x14 = Activation("relu")(x13)
        x15 = UpSampling2D()(x14)
        x16 = Conv2D(64, kernel_size=3, padding="same",
                     kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(x15)
        x17 = BatchNormalization(momentum=0.8)(x16)
        x18 = Activation("relu")(x17)
        x19 = UpSampling2D()(x18)
        x20 = Conv2D(32, kernel_size=3, padding="same")(x19)
        x21 = BatchNormalization(momentum=0.8)(x20)
        x22 = Activation("relu")(x21)
        x23= Conv2D(self.channels, kernel_size=3, padding="same",
                        kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(x22)
        output = Activation("tanh")(x23)
        gen = Model([word, img64], output)
        gen.summary()
        return gen
    def build_discriminator(self):
        word_1 = Input(shape=(describe_long,))
        emb=shared_embedder(word_1)
        img = Input(shape=self.img_shape)
        x1 = Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same")(img)
        x3 = LeakyReLU(alpha=0.2)(x1)
        x4 = Dropout(0.25)(x3)
        x5 = Conv2D(32, kernel_size=3, strides=2, padding="same")(x4)
        x6 = BatchNormalization(momentum=0.8)(x5)
        x7 = LeakyReLU(alpha=0.2)(x6)
        x8 = Dropout(0.25)(x7)
        x9 = Conv2D(64, kernel_size=3, strides=2, padding="same")(x8)
        x10 = BatchNormalization(momentum=0.8)(x9)
        x11 = LeakyReLU(alpha=0.2)(x10)
        x12 = Dropout(0.25)(x11)
        x13 = Conv2D(128, kernel_size=3, strides=2, padding="same")(x12)
        x14 = BatchNormalization(momentum=0.8)(x13)
        x15 = LeakyReLU(alpha=0.2)(x14)
        x16 = Dropout(0.25)(x15)
        x17 = Conv2D(256, kernel_size=3, strides=1, padding="same")(x16)
        x18 = BatchNormalization(momentum=0.8)(x17)
        x19 = LeakyReLU(alpha=0.2)(x18)
        x20 = Dropout(0.25)(x19)
        com=Concatenate()([emb,x19])
        x21 = Conv2D(512, kernel_size=3, strides=1, padding="same")(com)
        x22 = BatchNormalization(momentum=0.8)(x21)
        x23 = LeakyReLU(alpha=0.2)(x22)
        x24 = Dropout(0.25)(x23)
        x25 = Flatten()(x24)
        output = Dense(1, activation='sigmoid')(x25)
        validity = Model([word_1, img], output)
        validity.summary()
        return validity
    def train_on_batch(self,x_64, z_64,x_train, y_train,z_train,seed_=1):
        #x_64与z_64对应
        #x_train,y_train,z_train对应
        batch_size=x_64.shape[0]
        fake = np.zeros((batch_size, 1))
        vaule = np.ones((batch_size, 1))
        optimizer3 = Adam(0.0002, 0.5)
        gen = self.generator.predict([z_64, x_64])
        train_tag = np.concatenate((z_train, z_64), axis=0)
        train_img = np.concatenate((x_train, gen), axis=0)
        train_valid = np.concatenate((y_train, fake), axis=0)
        np.random.seed(seed_)
        np.random.shuffle(train_img)
        np.random.seed(seed_)
        np.random.shuffle(train_valid)
        np.random.seed(seed_)
        np.random.shuffle(train_tag)
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer3)
        d_loss = self.discriminator.train_on_batch([train_tag, train_img], train_valid)
        self.discriminator.trainable = False
        g_loss = self.combined.train_on_batch([z_64, x_64], vaule)
        print('256:',d_loss,g_loss)
    def save(self):
        self.discriminator.save('模型/disc256.keras')
        self.generator.save('模型/gene256.keras')
    def yuce(self, tag,img):
        if len(tag.shape) == 1:
            tag = np.expand_dims(tag, 0)
        gen_imgs = self.generator.predict([tag, img])  # 生成图像
        gen_imag = (gen_imgs + 1) * 127.5
        gen_image = np.round(gen_imag[0], decimals=0)
        gen_image = np.clip(gen_image, 0, 255).astype(np.uint8)#显示的图像
        return gen_image,gen_imgs

class Draw_model_512():
    def __init__(self):
        self.img_rows = 512
        self.img_cols = 512
        self.channels = 3
        # 图片长，宽，高
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        optimizer1 = Adam(0.0002, 0.5)
        optimizer2 = Adam(0.0002, 0.5)
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        try:
            self.discriminator = models.load_model('模型/disc512.keras')
            print('加载成功')
        except ValueError:
            print('加载失败')
        try:
            self.generator = models.load_model('模型/gene512.keras')
            print('加载成功')
        except ValueError:
            print('加载失败')
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer1, metrics=['accuracy'])
        z1 = Input(shape=(describe_long,))
        z2 = Input(shape=(256,256,3))
        img = self.generator([z1, z2])
        self.discriminator.trainable = False
        valid = self.discriminator([z1, img])
        self.combined = Model([z1, z2], valid)
        self.combined.compile(loss='mse', optimizer=optimizer2)
    def build_generator(self):
        # 生成图片256x256
        word = Input(shape=(describe_long,))
        img256 = Input(shape=(256,256,3))
        img256_1=Conv2D(32, kernel_size=3, strides=2, padding="same",
                       kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(img256)
        img256_2=Activation('relu')(img256_1)
        img256_3=Conv2D(64, kernel_size=3, strides=2, padding="same",
                       kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(img256_2)
        img256_4=BatchNormalization(momentum=0.8)(img256_3)
        img64_5=Activation('relu')(img256_4)
        img256_6 = Conv2D(128, kernel_size=3, strides=2, padding="same",
                         kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(img64_5)
        img256_7 = BatchNormalization(momentum=0.8)(img256_6)
        img64_8 = Activation('relu')(img256_7)
        img256_9 = Conv2D(256, kernel_size=3, strides=2, padding="same",
                         kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(img64_8)
        img256_10 = BatchNormalization(momentum=0.8)(img256_9)
        img256_11 = Activation('relu')(img256_10)
        word_1 = shared_embedder(word)
        com = Concatenate()([word_1,img256_11])
        x1 = Conv2D(1024, kernel_size=3, padding="same",
                    kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(com)
        x2 = Activation("relu")(x1)
        x3 = UpSampling2D()(x2)
        x4 = Conv2D(512, kernel_size=3, padding="same",
                    kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(x3)
        x5 = BatchNormalization(momentum=0.8)(x4)
        x6 = Activation("relu")(x5)
        x7 = UpSampling2D()(x6)
        x8 = Conv2D(256, kernel_size=3, padding="same",
                    kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(x7)
        x9 = BatchNormalization(momentum=0.8)(x8)
        x10 = Activation("relu")(x9)
        x11 = UpSampling2D()(x10)
        x12 = Conv2D(128, kernel_size=3, padding="same",
                     kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(x11)
        x13 = BatchNormalization(momentum=0.8)(x12)
        x14 = Activation("relu")(x13)
        x15 = UpSampling2D()(x14)
        x16 = Conv2D(64, kernel_size=3, padding="same",
                     kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(x15)
        x17 = BatchNormalization(momentum=0.8)(x16)
        x18 = Activation("relu")(x17)
        x19 = UpSampling2D()(x18)
        x20 = Conv2D(32, kernel_size=3, padding="same",
                     kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(x19)
        x21 = BatchNormalization(momentum=0.8)(x20)
        x22 = Activation("relu")(x21)
        x23= Conv2D(self.channels, kernel_size=3, padding="same",
                        kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(x22)
        output = Activation("tanh")(x23)
        gen = Model([word, img256], output)
        gen.summary()
        return gen
    def build_discriminator(self):
        word_1 = Input(shape=(describe_long,))
        word_2 = shared_embedder(word_1)
        img = Input(shape=self.img_shape)
        x1 = Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same")(img)
        x3 = LeakyReLU(alpha=0.2)(x1)
        x4 = Dropout(0.25)(x3)
        x5 = Conv2D(32, kernel_size=3, strides=2, padding="same")(x4)
        x6 = BatchNormalization(momentum=0.8)(x5)
        x7 = LeakyReLU(alpha=0.2)(x6)
        x8 = Dropout(0.25)(x7)
        x9 = Conv2D(64, kernel_size=3, strides=2, padding="same")(x8)
        x10 = BatchNormalization(momentum=0.8)(x9)
        x11 = LeakyReLU(alpha=0.2)(x10)
        x12 = Dropout(0.25)(x11)
        x13 = Conv2D(128, kernel_size=3, strides=2, padding="same")(x12)
        x14 = BatchNormalization(momentum=0.8)(x13)
        x15 = LeakyReLU(alpha=0.2)(x14)
        x16 = Dropout(0.25)(x15)
        x17 = Conv2D(256, kernel_size=3, strides=2, padding="same")(x16)
        x18 = BatchNormalization(momentum=0.8)(x17)
        x19 = LeakyReLU(alpha=0.2)(x18)
        x20 = Dropout(0.25)(x19)
        x21 = Conv2D(512, kernel_size=3, strides=1, padding="same")(x20)
        x22 = BatchNormalization(momentum=0.8)(x21)
        x23 = LeakyReLU(alpha=0.2)(x22)
        com=Concatenate()([word_2,x23])
        x20 = Flatten()(com)
        output = Dense(1, activation='sigmoid')(x20)
        validity = Model([word_1, img], output)
        validity.summary()
        return validity
    def train_on_batch(self,x_256, z_256,x_train, y_train,z_train,seed_=1):
        #x_256与z_256对应
        #x_train,y_train,z_train对应
        batch_size=x_256.shape[0]
        fake = np.zeros((batch_size, 1))
        vaule = np.ones((batch_size, 1))
        optimizer3 = Adam(0.0002, 0.5)
        gen = self.generator.predict([z_256, x_256])
        train_tag = np.concatenate((z_train, z_256), axis=0)
        train_img = np.concatenate((x_train, gen), axis=0)
        train_valid = np.concatenate((y_train, fake), axis=0)
        np.random.seed(seed_)
        np.random.shuffle(train_img)
        np.random.seed(seed_)
        np.random.shuffle(train_valid)
        np.random.seed(seed_)
        np.random.shuffle(train_tag)
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer3)
        d_loss = self.discriminator.train_on_batch([train_tag, train_img], train_valid)
        self.discriminator.trainable = False
        g_loss = self.combined.train_on_batch([z_256, x_256], vaule)
    def save(self):
        self.discriminator.save('模型/disc512.keras')
        self.generator.save('模型/gene512.keras')
    def yuce(self, tag,img):
        if len(tag.shape) == 1:
            tag = np.expand_dims(tag, 0)
        gen_imgs = self.generator.predict([tag, img])  # 生成图像
        gen_imag = (gen_imgs + 1) * 127.5
        gen_image = np.round(gen_imag[0], decimals=0)
        gen_image = np.clip(gen_image, 0, 255).astype(np.uint8)
        return gen_image
if __name__=='__main__':
    model_64 = Draw_model_64()
    model_256 = Draw_model_256()
    idx = np.random.randint(0, 64, 16)
    for each in range(1):
        for i in range(1):
            seed = randint(0, 30000)
            x64_train, y64_train, z64_train1, z64_train2 = 加载.load_batch('./64', 64)
            model_64.train_on_batch(x64_train, y64_train, z64_train1, seed)
            _, x_64 = model_64.yuce(z64_train1[idx])
            x256_train, y256_train, z256_train1, z256_train2 = 加载.load_batch('./256', 32)
            x_64 = np.concatenate((x_64, x64_train[idx]), axis=0)
            z_64 = np.concatenate((z64_train1[idx], z64_train2[idx]), axis=0)
            model_256.train_on_batch(x_64, z_64, x256_train, y256_train, z256_train1, seed)
        model_64.save()
        model_256.save()
        shared_embedder.save()
        _, pic = model_64.yuce(z256_train1[0])
        pic256, _ = model_256.yuce(z256_train1[0], pic)
        show_256 = Image.fromarray(pic256)
        show_256.save('./保存/测试/' + str((each + 1) * 1000) + '.png')
    p64=plt.figure()
    plt.plot(model_64.dis_losses,label='discriminator',color='red')
    plt.plot(model_64.gen_losses,label='generator',color='blue')
    plt.xlabel('epoch')
    plt.ylabel('discriminator loss')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1,1.05), loc='upper left')
    plt.show()
    '''
    x,y,z1,z2=加载.load_batch('./64', 1)
    img,_=model_64.yuce(z1)
    show_256=Image.fromarray(img)
    show_256.show()
    '''