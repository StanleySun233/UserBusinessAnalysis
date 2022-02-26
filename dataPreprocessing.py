import random as rd

import jieba as jb
import numpy as np
import pandas as pd


def getDataset():  # 36727是空的 删除
    data = pd.read_csv('./dataset/online_shopping_10_cats.csv')
    data = data[['cat', 'review']]
    print(len(data['cat']))

    countLabel = {'cat': data['cat'].value_counts().index,
                  'count': data['cat'].value_counts()}
    dataLabel = pd.DataFrame(data=countLabel).reset_index(drop=True)
    print(dataLabel)

    resLabel = pd.DataFrame(data=[countLabel['cat']], columns=range(10))
    resLabel.to_csv('./dataset/number2cat.csv', encoding='utf-8', index=False)

    x, y = list(data['review']), list(data['cat'])
    return x, y


def getNum2Cat():
    labelDict = {}
    data = pd.read_csv('./dataset/number2cat.csv', encoding='utf-8')

    for i in data.columns:
        labelDict[i] = data[i][0]

    return labelDict


def Cat2Number(labelDict):
    numberDict = dict(zip(labelDict.values(), labelDict.keys()))
    return numberDict


def getStopWord():
    resSheet = []
    f = open('./dataset/chineseStopWords.txt', 'r', encoding='utf-8').readlines()
    for i in f:
        resSheet.append(i.replace("\n", ""))
    return resSheet


def splitWord(data):
    resSheet = []
    for i in data:
        resSheet.append([])
        cut = jb.cut(i)
        for j in cut:
            resSheet[-1].append(j)

    f = open('./dataset/xSplitWord.txt', 'w', encoding='utf-8')

    for i in resSheet:
        writeString = ""
        for j in i:
            writeString += j + " "
        writeString = writeString[:-1] + "\n"
        f.write(writeString)
    f.close()
    return resSheet


def getSplitWord():
    resSheet = []
    f = open('./dataset/xSplitWord.txt', 'r', encoding='utf-8').readlines()
    for i in f:
        resSheet.append(i[:-1].split(" "))

    return resSheet


def splitWordWithoutStopWord(data, stopWord):
    resSheet = []
    for i in data:
        resSheet.append([])
        for j in i:
            if j not in stopWord:
                resSheet[-1].append(j)

    f = open('./dataset/xSplitWordWithoutStopWord.txt', 'w', encoding='utf-8')
    for i in resSheet:
        writeString = ""
        for j in i:
            writeString += j + " "
        writeString = writeString[:-1] + "\n"
        f.write(writeString)
    f.close()
    return resSheet


def getSplitWordWithoutStopWord():
    resSheet = []
    f = open('./dataset/xSplitWordWithoutStopWord.txt', 'r', encoding='utf-8').readlines()
    for i in f:
        resSheet.append(i[:-1].split(" "))
    return resSheet


def commentCut(data, lenSentense=100):  # you bug
    resSheet = []
    for i in data:
        resSheet.append(i[:max(lenSentense, len(i))])
    return resSheet


def word2NumDict(data):
    resSheet = []
    wordDict = {}
    l = len(data)
    cnt = 0
    for i in data:
        if cnt % 1000 == 0:
            print(cnt, l)
        for j in i:
            if j not in resSheet:
                resSheet.append(j)
        cnt += 1
    f = open('./dataset/xWord2Num.txt', 'w', encoding='utf-8')

    for i in range(len(resSheet)):
        f.write(resSheet[i] + ' ')
        wordDict[resSheet[i]] = i
    f.close()
    return wordDict


def getWord2NumDict():
    resSheet = {}
    f = open('./dataset/xWord2NumDict.txt', 'r', encoding='utf-8').readline().split(" ")[:-1]
    for i in range(len(f)):
        resSheet[f[i]] = i
    return resSheet


def translateWord2NumX(data, dataDict):
    resSheet = []

    for i in data:
        resSheet.append([])
        for j in i:
            resSheet[-1].append(dataDict[j])

    f = open('./dataset/xApterTranslate.txt', 'w', encoding='utf-8')
    for i in resSheet:
        writeString = ""
        for j in i:
            writeString += str(j) + " "
        writeString = writeString[:-1] + "\n"
        f.write(writeString)
    f.close()

    return resSheet


def translateWord2NumY(data, dataDict):
    resSheet = []

    for i in data:
        resSheet.append(i)

    f = open('./dataset/yApterPreprocessing.txt', 'w', encoding='utf-8')
    for i in resSheet:
        writeString = str(dataDict[i]) + " "
        writeString = writeString[:-1] + "\n"
        f.write(writeString)
    f.close()

    return resSheet


def getYAfterPreprocessing():
    resSheet = []
    f = open('./dataset/yApterPreprocessing.txt', 'r', encoding='utf-8')

    for i in f:
        resSheet.append(int(i[:-1]))
    f.close()

    return resSheet


def getXAfterTranslate():
    resSheet = []
    f = open('./dataset/xApterPreprocessing.txt', 'r', encoding='utf-8').readlines()
    for i in f:
        s = i.replace(" \n", "").split(" ")
        resSheet.append([])
        for j in s:
            resSheet[-1].append(int(j))
    return resSheet


def setSentense2Lenght(data, dataLen=100):
    resSheet = []
    for i in data:
        resSheet.append(i)
        for j in range(len(i), dataLen):
            resSheet[-1].append(0)

    f = open('./dataset/xApterPreprocessing.txt', 'w', encoding='utf-8')
    for i in resSheet:
        writeString = ""
        for j in i:
            writeString += str(j) + " "
        writeString = writeString[:-1] + "\n"
        f.write(writeString)
    f.close()

    return resSheet


def getXAfterPreprocessing():
    resSheet = []
    f = open('./dataset/xApterPreprocessing.txt', 'r', encoding='utf-8').readlines()
    for i in f:
        s = i.replace(" \n", "").split(" ")
        resSheet.append([])
        for j in s:
            resSheet[-1].append(int(j))
    return resSheet


def randomizeDataaset(x, y, randomTestSize=0.3):
    xTrain = []
    yTrain = []
    xTest = []
    yTest = []

    for i in range(len(x)):
        seed = rd.randint(0, 101)
        if seed < randomTestSize * 100:
            xTest.append(x[i][:100])
            yTest.append(int(y[i]))
        else:
            xTrain.append(x[i][:100])
            yTrain.append(int(y[i]))
    return np.array(xTrain), np.array(yTrain), np.array(xTest), np.array(yTest)
