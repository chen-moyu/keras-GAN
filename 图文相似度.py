import pickle
import os
import tkinter as tk
from random import choice
from PIL import Image, ImageTk
import tkinter.font as tkFont
import init
windows=init.root
font1=tkFont.Font(family='微软雅黑',size=14)
with open('标签索引.pkl', 'rb') as f:
    sequence_number = pickle.load(f)  # sequence为字典{标签中文件名：数字}
with open('相似度表.pkl', 'rb') as f:
    form = pickle.load(f)  # 二维列表
file = os.listdir('./数字标签')
row,column=0,0
def choice_image_text():
    global row,column
    while True:
        img = choice(file)  # 随机选择图片
        tag = choice(file)  # 随机选择文本
        row = sequence_number[img]
        column = sequence_number[tag]
        if form[row][column] ==0:
            break
    with open('./512/'+img[0:-4], 'rb') as f:
        image = pickle.load(f)
        image = Image.fromarray(image)
        image=image.resize((210,210))
        image = ImageTk.PhotoImage(image)
    with open('./标签/'+tag, 'rb') as f:
        text=pickle.load(f)
    return image,text
def next_image_text():
    #修改相似度，并切换下一组
    form[row][column]=float(similar_entry.get())
    image,text=choice_image_text()
    image_show.config(image=image)
    text_show.delete('0.0',tk.END)
    text_show.insert('0.0',text)
def save_quit():
    with open('相似度表.pkl', 'wb') as f:
        pickle.dump(form,f)
    windows.quit()
image,text=choice_image_text()
image_show=tk.Label(windows,image=image)
text_show=tk.Text(windows,width=23,height=10,font=font1)
text_show.insert('0.0',text)
save_button=tk.Button(windows,text='保存并退出',fg='green',command=save_quit)
similar=tk.Label(windows,text='相似度[0,1]')
similar_entry=tk.Entry(windows)
comfirm_button=tk.Button(windows,text='确认',command=next_image_text,width=10)
def turn_this():
    image_show.grid(row=1, column=0, columnspan=2)
    text_show.grid(row=1, column=2, columnspan=2)
    save_button.grid(row=2, column=0, pady=(10, 15))
    similar.grid(row=2, column=1, pady=(10, 15), padx=(10, 0))
    similar_entry.grid(row=2, column=2, pady=(10, 15), padx=(0, 10))
    comfirm_button.grid(row=2, column=3, pady=(10, 15))
def destroy_self():
    image_show.grid_remove()
    text_show.grid_remove()
    save_button.grid_remove()
    similar.grid_remove()
    similar_entry.grid_remove()
    comfirm_button.grid_remove()
if __name__ == '__main__':
    turn_this()
    windows.mainloop()