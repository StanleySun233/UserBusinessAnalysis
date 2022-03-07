import seaborn as sns
import tensorflow.keras as keras
from matplotlib import pyplot as plt
from sklearn.metrics import *
from sklearn.metrics import classification_report

import dataPreprocessing as dp

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
x, y = dp.getXAfterPreprocessing(), dp.getYAfterPreprocessing()
xTrain, yTrain, xTest, yTest = dp.randomizeDataset(x, y)
yTrain = keras.utils.to_categorical(yTrain)
yTest = keras.utils.to_categorical(yTest)
Y_test = yTest
model = keras.models.load_model('model-LSTM.h5')

y_pred = model.predict(xTest)
y_pred = y_pred.argmax(axis=1)
Y_test = Y_test.argmax(axis=1)
yDict = dp.cat2Number(dp.getNum2Cat())
yDict = yDict.keys()

conf_mat = confusion_matrix(Y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=yDict, yticklabels=yDict)
ax.xaxis.set_ticks_position("top")
plt.ylabel('实际结果', fontsize=18)
plt.xlabel('预测结果', fontsize=18)
plt.show()

print('accuracy %s' % accuracy_score(y_pred, Y_test))
print(classification_report(Y_test, y_pred, target_names=yDict))
