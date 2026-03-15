#
import ollama
import os
import pickle
client = ollama.Client(host='http://127.0.0.1:11434')
def describe_image(image_path):
    try:
        # 构建 Ollama prompt
        prompt1 = "描述这张图片"
        # 调用 Ollama API
        response1 = client.chat(
            model="gemma3:4b",
            messages=[
                {
                    "role": "user",
                    "content": prompt1,
                    "images":[image_path,]
                },
            ],
            stream=False  # 不使用流式传输，获取完整结果
        )
        label = response1["message"]["content"]
        label=label.replace("*","")
        label=label.replace(" ","")
        label=label.replace("\n","")
        return label
    except FileNotFoundError:
        print(f"文件未找到: {image_path}")
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        return None
def describe_all():
    #ai标注所有图片
    a=os.listdir('./数据')
    b=os.listdir('./标签')
    for i in b:
        a.remove(i[0:-4])
    for i in a:
        text=describe_image('./数据/'+i)
        c=open('./标签/'+i+'.pkl','wb')
        pickle.dump(text,c)
        c.close()
        print(i,'已标注')
    d=[]
    e=open('已检查.pkl','wb')
    pickle.dump(d,e)
    e.close()
if __name__ == "__main__":
    '''
    image_file_path = r"D:\\画图ai2.0\数据\00-00-05-dc9736a308543b1a354c49fd664e508f003ae678.png@656w_548h_progressive.webp"  # 替换为你的图片路径
    keywords = describe_image(image_file_path)
    print(keywords)
    '''
    describe_all()