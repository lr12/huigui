from DealtextUtil import DealtextUtil
import numpy as np
import os
import gensim
import pandas as pd

def get_vec_data(path):
    dtu=DealtextUtil()
    y= dtu.readdata("./data/predata.csv")
    y = y.values
    # print()
    y=y[:,1:]

    x=getvecmatrix()
    x=x.reshape(-1,300)
    print(x.shape)
    if len(x)!=len(y):
        print("出错了出错了")
    return x,y
#这里的路径需要改
def getvecmatrix():
    pathlist=['./退款','./未退款']
    data = []
    dtu = DealtextUtil()
    for path in pathlist:
        filelist = os.listdir(path)
        for file in filelist:
            filepath = os.path.join(path, file)
            words = dtu.readdoc_word(filepath)

            print(words)
            data.append(words)
    matrix=data2vec(data)
    return matrix

def data2vec(datas):
    model = get_model()
    matrix=[]
    for data in datas:
        wvec_t = np.zeros(([1, 300]))
        for word in data:
            try:
                wvec = model[word]
            except:
                wvec = np.random.uniform(-0.25, 0.25, 300)
            wvec_t+=wvec
        wvec_t=wvec/len(data)
        wvec_t = np.array(wvec_t).reshape(1, 300)
        matrix.append(wvec_t)
    matrix=np.array(matrix)
    return matrix


def get_model():
    model = gensim.models.Word2Vec.load('./vecmodel/zh.bin')
    return model
