#
import tkinter as tk
from tkinter import ttk
import numpy as np
import 模型
import 加载
from PIL import ImageTk, Image
import pickle
import os
import init
from random import randint
import tkinter.font as tkFont

windows = init.root
font1=tkFont.Font(family='微软雅黑',size=14)
def show_img(img):
    #展示生成的图片
    global gen_img
    gen_image = (img + 1) * 127.5
    gen_image = np.round(gen_image[0], decimals=0)
    gen_image = np.clip(gen_image, 0, 255).astype(np.uint8)
    image = Image.fromarray(gen_image)
    image = ImageTk.PhotoImage(image)
    generate_show.configure(image=image)
gen_img=Image.open('./背景.png')
gen_img=gen_img.convert('RGB')
gen_img=gen_img.resize((210,210))
gen_img=ImageTk.PhotoImage(gen_img)
generate_show=tk.Label(windows,image=gen_img)
text_entry=tk.Text(windows,width=20,height=8,font=font1)
generate_show.grid(row=1,column=2,columnspan=2)
text_entry.grid(row=1,column=0,columnspan=2)
def save_image():
    #保存生成的图片
    a=os.listdir('./保存')
    b=str(len(a))
    global gen_img
    gen_img.save('./保存/'+b+'.png')
save_button=tk.Button(windows,text='保存图片',command=save_image)
save_button.grid(row=2,column=3)

model_64=模型.Draw_model_64()
model_256=模型.Draw_model_256()
model_512=模型.Draw_model_512()
var256=tk.IntVar()
var512=tk.IntVar()
generate_256=ttk.Checkbutton(windows,text='256x256',variable=var256)
generate_512=ttk.Checkbutton(windows,text='512x512',variable=var512)
generate_256.grid(row=2,column=0)
generate_512.grid(row=2,column=1)
def gan_generate():
    global gen_img
    text=text_entry.get('0.0',tk.END)
    text=加载.word_to_number_every(text)
    image,img_64=model_64.yuce(text)
    if var256.get():
        image,img_256=model_256.yuce(text,img_64)
    if var512.get() and var256.get():
        image=model_512.yuce(text,img_256)
    image=Image.fromarray(image)
    gen_img=image.copy()
    image.show()
    image = image.resize((210, 210))
    image = ImageTk.PhotoImage(image)
    generate_show.config(image=image)
def train(epochs,test_epoch=1,batch_size=64,test=False):
    idx = np.random.randint(0, 64, batch_size//4)
    for each in range(epochs//test_epoch):
        for i in range(test_epoch):
            seed = randint(0, 30000)
            x64_train, y64_train, z64_train1, z64_train2 = 加载.load_batch('./64', batch_size)
            model_64.train_on_batch(x64_train, y64_train, z64_train1, seed)
            _, x_64 = model_64.yuce(z64_train1[idx])
            x256_train, y256_train, z256_train1, z256_train2 = 加载.load_batch('./256', batch_size//2)
            x_64 = np.concatenate((x_64, x64_train[idx]), axis=0)
            z_64 = np.concatenate((z64_train1[idx], z64_train2[idx]), axis=0)
            model_256.train_on_batch(x_64, z_64, x256_train, y256_train, z256_train1, seed)
            train_progress['value'] = (each+1)*(i+1)/epochs
            windows.update()
        model_64.save()
        model_256.save()
        模型.shared_embedder.save()
        if test:
            _, pic = model_64.yuce(z256_train1[0])
            pic256, _ = model_256.yuce(z256_train1[0], pic)
            show_256 = Image.fromarray(pic256)
            show_256.save('./保存/测试/' + str((each + 1) * test_epoch) + '.png')
def start_train():
    epoch=int(epoch_entry.get())
    batch=int(batch_entry.get())
    train(epoch,batch_size=batch)
gan_generate_button=tk.Button(windows,text='生成',command=gan_generate)
gan_generate_button.grid(row=2,column=2)
#以下代码为训练
epoch_lable=tk.Label(windows,text='训练轮次')
epoch_entry=tk.Entry(windows)
start_train=tk.Button(windows,text='开始训练',command=start_train)
train_progress=ttk.Progressbar(windows)
train_progress['maximum'] = 1
train_progress['value'] = 0
epoch_lable.grid(row=3,column=0)
epoch_entry.grid(row=3,column=1)
start_train.grid(row=3,column=2)
train_progress.grid(row=3,column=3)
batch_lable=tk.Label(windows,text='批次大小')
batch_entry=tk.Entry(windows)
batch_entry.insert(0,'64')
batch_lable.grid(row=4,column=0)
batch_entry.grid(row=4,column=1)
vartest=tk.IntVar()
is_test=ttk.Checkbutton(windows,text='测试',variable=vartest)
if __name__ == '__main__':
    windows.mainloop()