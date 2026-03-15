import pickle
import os
import json
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from random import choice,sample
from PIL import Image
def create_tokenizer():
    #创建文字数字编码表
    word = []
    path = os.listdir('./标签')
    for i in path:
        a = open('./标签/'+i, 'rb')
        b = pickle.load(a)
        for j in b:
            if j not in word:
                word.append(j)
        a.close()
    tokenizer = {word[i]:i+1 for i in range(len(word))}
    t = open('tokenizer.json', 'w')
    json.dump(tokenizer, t)
    t.close()
    return tokenizer
try:
    t=open('tokenizer.json','r')
    tokenizer=json.load(t)
    t.close()
except FileNotFoundError:
    tokenizer=create_tokenizer()
def word_to_number_all():
    #将所有的文本标签转为数字标签并保存
    word=[]#元素为数字
    path = os.listdir('./标签')
    path2=os.listdir('./512')
    for i in path:
        if i[0:-4] not in path2:
            continue
        a = open('./标签/' + i, 'rb')
        b = pickle.load(a)
        for j in b:
            try:
                word.append(tokenizer[j])
            except KeyError:
                print('字典未收录：',j)
        a.close()
        c=open('./数字标签/'+i,'wb')
        pickle.dump(word, c)
        word.clear()
def word_to_number_every(word):
    #将单个文本转为数字,word为文本
    text=[[]]
    for i in word:
        try:
            text[0].append(tokenizer[i])
        except KeyError:
            print('字典未收录：', i)
    text=np.array(text)
    text = pad_sequences(text, maxlen=540, padding='post')
    return text
def load_batch(path,n):
    x_train=[]#图片
    y_train=[]#判别器输出
    z1_train=[]#随机文字描述
    z2_train=[]#原本文字描述
    with open('标签索引.pkl', 'rb') as f:
        sequence_number = pickle.load(f)  # sequence为字典{标签中文件名：数字}
    with open('相似度表.pkl', 'rb') as f:
        form = pickle.load(f)  # 二维列表
    file=os.listdir('./数字标签')
    for i in range(n):
        tag_real=choice(file)#选择图片和对应的文本
        tag_fake=choice(file)#随机选择的文本
        row=sequence_number[tag_real]
        column=sequence_number[tag_fake]
        with open(path+'/'+tag_real[0:-4],'rb') as f:
            image=pickle.load(f)
        x_train.append(image)
        y_train.append([form[row][column]])
        with open('./数字标签/'+tag_real,'rb') as f:
            tag_r=pickle.load(f)
        with open('./数字标签/'+tag_fake,'rb') as f:
            tag_f=pickle.load(f)
        z1_train.append(tag_f)
        z2_train.append(tag_r)
    z1_train = pad_sequences(z1_train, maxlen=540, padding='post')
    z2_train = pad_sequences(z2_train, maxlen=540, padding='post')
    x_train = np.array(x_train)
    x_train = x_train / 127.5 - 1.
    y_train = np.array(y_train)
    z1_train = np.array(z1_train)
    z1_train = z1_train.astype(np.int32)
    z2_train = np.array(z2_train)
    z2_train = z2_train.astype(np.int32)
    return x_train, y_train, z1_train, z2_train
def load_original_image():
    #加载原始图片
    file=os.listdir('./数据')
    x_train=[]
    img = Image.open('./数据/'+choice(file))
    size=img.size
    if size[0]>1600 and size[1]>1600:
        if size[0]>size[1]:
            x=1600
            y=size[1]/size[0]*x
        else:
            y=1600
            x=size[0]/size[1]*y
        size=[int(x),int(y)]
        img=img.resize(size)
    if size[0] % 8 != 0 or size[1] % 8 != 0:
        img=img.resize((size[0]//8*8,size[1]//8*8))
    img = img.convert('RGB')
    img = np.array(img)
    x_train.append(img)
    x_train=np.array(x_train)
    x_train=x_train / 127.5 - 1.
    return x_train
if __name__=='__main__':
    '''
    x,y,z1,z2=load_batch('./64',16)
    print(x.shape)
    print(y)
    print(z1.shape)
    '''
    cs=word_to_number_every('测试，蓝天白云，阳光灿烂')
    print(cs)