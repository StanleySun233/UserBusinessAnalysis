import tensorflow as tf
import tensorflow.keras as keras
from keras.callbacks import *

import dataPreprocessing as dp
import model as md
import rewriteCallbak as rc

print(tf.config.list_physical_devices('GPU'))
x, y = dp.getXAfterPreprocessing(), dp.getYAfterPreprocessing()
xTrain, yTrain, xTest, yTest = dp.randomizeDataaset(x, y)

yTrain, yTest = keras.utils.to_categorical(yTrain), keras.utils.to_categorical(yTest)

print(np.shape(xTrain))
print(np.shape(yTrain))
print(np.shape(xTest))
print(np.shape(yTest))

model = md.modelLSTM(xTrain.shape[1])
epoch = 50
batch_size = 32

history = rc.LossHistory("LSTM")

model.fit(xTrain, yTrain, epochs=epoch, batch_size=batch_size, validation_data=(xTest, yTest),
          callbacks=[history])

model.save("model-LSTM.h5")

history.write()
