import dataPreprocessing as dp

X, Y = dp.getDataset()
stop = dp.getStopWord()

X = dp.splitWord(X)
X = dp.splitWordWithoutStopWord(X, stop)
X = dp.commentCut(X)

XDict = dp.word2NumDict(X)
YDict = dp.Cat2Number(dp.getNum2Cat())

X = dp.translateWord2NumX(X, XDict)
Y = dp.translateWord2NumY(Y, YDict)

X = dp.setSentense2Lenght(X)
