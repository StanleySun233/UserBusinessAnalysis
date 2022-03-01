import tensorflow.keras as keras


def modelLSTM(shape):
    model = keras.Sequential([
        keras.layers.Embedding(50000, 256, input_length=shape),
        keras.layers.SpatialDropout1D(0.2),
        keras.layers.LSTM(256),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model


def modelCNN(shape):
    model = keras.Sequential([
        keras.layers.Embedding(50000, 256, input_length=shape),
        keras.layers.GlobalAvgPool1D(),
        keras.layers.SpatialDropout1D(0.2),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(10, activation='softmax')])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model
