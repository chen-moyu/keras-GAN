import tkinter as tk
import init
import 标注
import 准备
import 图文相似度
page=1 #窗口页面，0 使用和训练，1 标注，2 准备，3 图文相似度
root=init.root
def turn_page(n):
    #切换页面，n为页面代号
    global page
    if page==0:
        pass
    elif page==1:
        标注.destroy_self()
    elif page==2:
        准备.destroy_self()
    elif page==3:
        图文相似度.destroy_self()
    page=n
    if n==0:
        pass
    elif n==1:
        标注.turn_this()
    elif n==2:
        准备.turn_this()
    elif n==3:
        图文相似度.turn_this()
use_button=tk.Button(root,text='使用和训练',command=lambda :turn_page(0))
mark_button=tk.Button(root,text='标注',command=lambda :turn_page(1))
prepare_button=tk.Button(root,text='准备',command=lambda :turn_page(2))
similar_button=tk.Button(root,text='图文相似度',command=lambda :turn_page(3))
use_button.grid(row=0,column=0)
mark_button.grid(row=0,column=1)
prepare_button.grid(row=0,column=2)
similar_button.grid(row=0,column=3)
root.mainloop()