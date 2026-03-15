#
from PIL import Image,ImageTk
import tkinter as tk
from tkinter import filedialog
import pickle
import os
import numpy as np
import ai标注
import tkinter.font as tkFont
import init
root=init.root
font1=tkFont.Font(family='微软雅黑',size=14)
class Picture():
    def __init__(self,path):
        self.path=path
        self.show_size=(210,210)
        _, self.name = os.path.split(self.path)
        self.image=Image.open(path)
        self.image=self.image.convert('RGB')
        self.imaged=self.image.copy()#原始图片副本
        self.size=self.image.size
        if self.size[0]>self.size[1]:
            self.position=[0,0,self.size[1],self.size[1]]
        else:
            self.position=[0,0,self.size[0],self.size[0]]
        self.positiond=self.position[:]
        self.images=self.image.crop(self.position)
        self.img_show=self.images.resize(self.show_size)
        self.img=ImageTk.PhotoImage(self.img_show)
    def move(self,x1,y1,x2,y2):
        #图片移动
        self.position[0]+=x1
        self.position[1]+=y1
        self.position[2]+=x2
        self.position[3]+=y2
        if self.position[2]>self.size[0] or self.position[0]<0:
            self.position[0]-=x1
            self.position[2]-=x2
        if self.position[3]>self.size[1] or self.position[1]<0:
            self.position[1]-=y1
            self.position[3]-=y2
        self.images = self.image.crop(self.position)
        self.img_show = self.images.resize(self.show_size)
        self.img = ImageTk.PhotoImage(self.img_show)
    def fill(self,x):
        #用白色填充图片窄边,x为比例,新图片窄边=原图片窄边*(1+2x)
        if self.size[0]!=self.size[1]:
            if self.size[0]>self.size[1]:
                fill_size=int(self.size[1]*x)
                if 2*fill_size+self.size[1]>self.size[0]:
                    fill_size=(self.size[0]-self.size[1])//2
                background=Image.new('RGB',(self.size[0],self.size[1]+2*fill_size),'white')
                background.paste(self.imaged,(0,fill_size))
            else:
                fill_size=int(self.size[0]*x)
                if 2*fill_size+self.size[0]>self.size[1]:
                    fill_size=(self.size[1]-self.size[0])//2
                background=Image.new('RGB',(self.size[0]+2*fill_size,self.size[1]),'white')
                background.paste(self.imaged,(fill_size,0))
            self.position[2]=self.positiond[2]+2*fill_size
            self.position[3]=self.positiond[3]+2*fill_size
            self.image=background
            self.images = self.image.crop(self.position)
            self.img_show = self.images.resize(self.show_size)
            self.img = ImageTk.PhotoImage(self.img_show)
    def save(self):
        #图片保存为numpy数组64，128，256，512大小各存一份
        image_64=self.images.resize((64,64))
        image_128=self.images.resize((128,128))
        image_256=self.images.resize((256,256))
        image_512=self.images.resize((512,512))
        image_64=np.array(image_64)
        image_128=np.array(image_128)
        image_256=np.array(image_256)
        image_512=np.array(image_512)
        d=open("./64/"+self.name,"wb")
        pickle.dump(image_64,d)
        a=open("./128/"+self.name,"wb")
        pickle.dump(image_128,a)
        b=open("./256/"+self.name,'wb')
        pickle.dump(image_256,b)
        c=open("./512/"+self.name,"wb")
        pickle.dump(image_512,c)
        a.close()
        b.close()
        c.close()
        d.close()
directory_path = './数据'
files = os.listdir(directory_path)
filed=os.listdir('./标签')
if '已检查.pkl' in os.listdir('.'):
    checks=True
    checked=open('已检查.pkl','rb')
    check=pickle.load(checked)
    checked.close()
    for i in check:
        files.remove(i)
else:
    checks = False
    for each in filed:
        files.remove(each[0:-4])
picture=Picture('./数据/'+files[0])
screen = tk.Label(root, image=picture.img)
def up():
    picture.move(0,10,0,10)#上移
    screen.config(image=picture.img)
def down():
    picture.move(0,-10,0,-10)#下移
    screen.config(image=picture.img)
def left():
    picture.move(10,0,10,0)#左移
    screen.config(image=picture.img)
def right():
    picture.move(-10,0,-10,0)#右移
    screen.config(image=picture.img)
def save_image(img=True,tag=True):
    #保存图片和相应描述
    global picture
    if checks:
        check.append(picture.name)
    if tag:
        a=open('./标签/'+picture.name+'.pkl','wb')
        pickle.dump(text.get('0.0',tk.END),a)
        a.close()
        print(picture.name,'描述已保存')
    if img:
        picture.save()
        print(picture.name,'图片已保存')
    files.remove(files[0])
    picture = Picture('./数据/' + files[0])
    screen.config(image=picture.img)
    text.delete('0.0', tk.END)
    fill_entry.delete(0, tk.END)
    load_text()
def out():
    #退出
    if checks:
        a=open('已检查.pkl','wb')
        pickle.dump(check,a)
    root.quit()
text=tk.Text(root,height=10,width=23,font=font1)
def load_text():
    #加载图片描述
    if picture.name+'.pkl' in filed:
        a=open('./标签/'+picture.name+'.pkl','rb')
        b=pickle.load(a)
        text.delete('0.0',tk.END)
        text.insert('0.0',b)
        print("描述加载成功")
    else:
        print("描述加载失败")
load_text()
def max_img():
    picture.images.show()
def chaxun_tag():
    file=os.listdir('./512')
    global picture
    tag=filedialog.askopenfilename()
    picture=Picture(tag)
    _, name = os.path.split(tag)
    if name in file:
        a=open('./512/'+name,'rb')
        img=pickle.load(a)
        img=Image.fromarray(img)
        img.show()
    screen.config(image=picture.img)
    load_text()
def fill_image():
    a=float(fill_entry.get())
    picture.fill(a)
    screen.config(image=picture.img)
def ai_mark():
    a=ai标注.describe_image('./数据/'+picture.name)
    text.insert('0.0',a)
def abandon_text():
    #撤销修改描述，恢复原本描述
    load_text()
def del_image():
    #删除图片及相关描述
    global picture
    os.remove('./数据/'+picture.name)
    a=os.listdir('./64')
    if picture.name in a:
        os.remove('./64/'+picture.name)
        os.remove('./128/' + picture.name)
        os.remove('./256/' + picture.name)
        os.remove('./512/' + picture.name)
    b=os.listdir('./标签')
    if picture.name+'.pkl' in b:
        os.remove('./标签/'+picture.name+'.pkl')
    picture = Picture('./数据/' + files[0])
    screen.config(image=picture.img)
    text.delete('0.0', tk.END)
    fill_entry.delete(0, tk.END)
    load_text()
mu=tk.Button(root,text="上",command=up,width=11,height=1)
md=tk.Button(root,text="下",command=down,width=11,height=1)
ml=tk.Button(root,text="左",command=left,width=11,height=1)
mr=tk.Button(root,text="右",command=right,width=11,height=1)
chaxun=tk.Button(root,text='打开指定图片',command=chaxun_tag,width=11,height=1)
max_i=tk.Button(root,text='大图',command=max_img,width=11,height=1)
ai_describe=tk.Button(root,text='ai描述',command=ai_mark,width=11, height=1)
delete_button=tk.Button(root,text='删除图片',command=del_image,width=11,height=1,fg='red')
abandon_text_button=tk.Button(root,text='撤销修改描述',command=abandon_text,width=11,height=1)
save=tk.Button(root,text='保存图片和描述',command=save_image,width=11,height=1)
save_img=tk.Button(root,text='仅保存图片',command=lambda: save_image(tag=False),width=11,height=1)
save_test=tk.Button(root,text='仅保存描述',command=lambda: save_image(img=False),width=11,height=1)
tc=tk.Button(root,text="退出",command=out,width=11,height=1,fg='red')
fill_text=tk.Label(root,text='填充比例(0~1)')
fill_entry=tk.Entry(root,width=12)
fill_button=tk.Button(root,text='填充',command=fill_image,width=11,height=1)
def turn_this():
    screen.grid(row=1, column=0, columnspan=2)
    text.grid(row=1, column=2, columnspan=2)
    mu.grid(row=2, column=0, pady=(25, 10))
    md.grid(row=2, column=1, pady=(25, 10))
    ml.grid(row=2, column=2, pady=(25, 10))
    mr.grid(row=2, column=3, pady=(25, 10))
    chaxun.grid(row=3, column=0)
    max_i.grid(row=3, column=1)
    ai_describe.grid(row=3, column=2)
    delete_button.grid(row=3, column=3)
    abandon_text_button.grid(row=4, column=0, pady=(10, 0))
    save.grid(row=4, column=1, pady=(10, 0))
    save_img.grid(row=4, column=2, pady=(10, 0))
    save_test.grid(row=4, column=3, pady=(10, 0))
    fill_text.grid(row=5, column=0, pady=(10, 15))
    fill_entry.grid(row=5, column=1, pady=(10, 15))
    fill_button.grid(row=5, column=2, pady=(10, 15))
    tc.grid(row=5, column=3, pady=(10, 15))
def destroy_self():
    screen.grid_remove()
    text.grid_remove()
    mu.grid_remove()
    md.grid_remove()
    ml.grid_remove()
    mr.grid_remove()
    chaxun.grid_remove()
    max_i.grid_remove()
    ai_describe.grid_remove()
    delete_button.grid_remove()
    abandon_text_button.grid_remove()
    save.grid_remove()
    save_img.grid_remove()
    save_test.grid_remove()
    fill_text.grid_remove()
    fill_entry.grid_remove()
    fill_button.grid_remove()
    tc.grid_remove()
if __name__=='__main__':
    turn_this()
    root.mainloop()