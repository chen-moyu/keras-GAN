#
import tkinter as tk
import ai标注
import os
import pickle
import 加载
import init
from tkinter import messagebox
root = init.root
def create_file():
    path=['64','128','256','512','保存/测试','数据','模型','标签','数字标签']
    for i in path:
        if not os.path.exists(i):
            os.makedirs(i)
def max_tag():
    #统计最长的文本
    m=0
    filed = os.listdir('./标签')
    for each in filed:
        j = open('./标签/' + each, 'rb')
        k = pickle.load(j)
        j.close()
        a=len(k)
        if a>m:
            m=a
    messagebox.showinfo(message='最长文本长度：'+str(m))
def creat_form():
    #创建索引和相似度表
    file=os.listdir('./标签')
    file2=os.listdir('./512')
    if '标签索引.pkl' in os.listdir('.'):
        with open('标签索引.pkl','rb') as f:
            sequence_number=pickle.load(f)#sequence为字典{标签中文件名：数字}
        num=len(sequence_number)
        for i in file:
            if i not in sequence_number and i in file2:
                sequence_number[i]=num
                num+=1
        with open('相似度表.pkl','rb') as f:
            form=pickle.load(f)#二维列表[图片][文本]
        new=[0]*(num-len(form[0]))
        for i in form:
            i.extend(new)
        for i in sequence_number:
            row=sequence_number[i]
            column=row
            form[row][column]=1
    else:
        num=len(file2)
        sequence_number={file2[i]+'.pkl':i for i in range(num)}
        with open('标签索引.pkl','wb') as f:
            pickle.dump(sequence_number,f)
        form=[[1 if i==j else 0 for i in range(num)] for j in range(num)]
        print(form)
        with open('相似度表.pkl','wb') as f:
            pickle.dump(form,f)
create_file_button=tk.Button(root,text='创建目录',command=create_file,width=10)
create_token=tk.Button(root,text='创建token',command=加载.create_tokenizer, width=10)
max_button=tk.Button(root,text='最长文本长度',command=max_tag,width=10)
ai_mark=tk.Button(root,text='AI预标注',command=ai标注.describe_all,width=10)
create_form_button=tk.Button(root,text='创建相似度表',command=creat_form,width=10)
turn_number=tk.Button(root,text='文本转数字',command=加载.word_to_number_all,width=10)
def turn_this():
    create_file_button.grid(row=1,column=0,pady=(10,15),padx=(10,10))
    create_token.grid(row=1,column=1,pady=(10,15),padx=(10,10))
    max_button.grid(row=2,column=0,padx=(10,10))
    ai_mark.grid(row=2,column=1)
    create_form_button.grid(row=3,column=0,pady=(15,15),padx=(10,10))
    turn_number.grid(row=3,column=1,padx=(10,10),pady=(15,15))
def destroy_self():
    create_file_button.grid_remove()
    create_token.grid_remove()
    max_button.grid_remove()
    ai_mark.grid_remove()
    create_form_button.grid_remove()
    turn_number.grid_remove()
if __name__ == '__main__':
    turn_this()
    root.mainloop()
