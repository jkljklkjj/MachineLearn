window_size = 7
fea_num = 5
 
 
model = keras.models.Sequential([
    keras.layers.Input((window_size, fea_num)),
    keras.layers.Reshape((window_size, fea_num, 1)),
    keras.layers.Conv2D(filters=64,
                           kernel_size=3,
                           strides=1,
                           padding="same",
                           activation="relu"),
    keras.layers.MaxPooling2D(pool_size=2, strides=1, padding="same"),
    keras.layers.Dropout(0.3),
    keras.layers.Reshape((window_size, -1)),
    keras.layers.LSTM(128, return_sequences=True),
    keras.layers.LSTM(64, return_sequences=False),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1)
])
 
 
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.summary()