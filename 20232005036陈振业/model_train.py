import model_build

def model_train(X_train, y_train, X_test, y_test):
    '''
    输入: 构建后的模型和股票数据
    输出: 训练模型的历史记录
    '''    
    from keras.callbacks import EarlyStopping
    #早停策略
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    
    model = model_build.model_build(X_train)

    history = model.fit(
        X_train,
        y_train,
        epochs=40,
        batch_size=32,
        validation_data=(X_test, y_test),
        shuffle=False,
        callbacks=[es]
    )
    model.save('股票数据预测.h5')
    return history