def model_build(X_train):
    from keras.models import Sequential
    from keras.optimizers import Nadam
    from keras.layers import Conv1D, LSTM, Dense, Dropout, Input

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu"
        ),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(5)
    ])

    model.compile(optimizer=Nadam(learning_rate=0.001), loss="mse")
    return model