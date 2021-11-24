if __name__ == "__main__":
    import numpy as np
    np.random.seed(42)
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import InputLayer
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    tf.random.set_seed(42)

    # BN parameters
    batch_size = 64
    print("batch_size = "+str(batch_size))
    alpha = .1
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))

    # MLP parameters
    num_units = 4192
    print("num_units = "+str(num_units))
    n_hidden_layers = 1
    print("n_hidden_layers = "+str(n_hidden_layers))

    # Training parameters
    num_epochs = 1000
    print("num_epochs = "+str(num_epochs))

    # Dropout parameters
    dropout_in = .2 # 0. means no dropout
    print("dropout_in = "+str(dropout_in))
    dropout_hidden = .5
    print("dropout_hidden = "+str(dropout_hidden))

    # LR
    LR = .001
    print("LR = "+str(LR))
    
    # Patience
    patience = 1
    print("Patience = "+str(patience))
    
    # Model Ouput Paths
    model_path = "mlp_model4.json"
    print("model_path = "+str(model_path))
    
    weights_path = "mlp_weights4.h5"
    print("weights_path = "+str(weights_path))

    print('Loading dataset...')

    # Load data
    train = np.loadtxt('train4.csv', delimiter=',', skiprows=1)
    valid = np.loadtxt('valid4.csv', delimiter=',', skiprows=1)
    test = np.loadtxt('test4.csv', delimiter=',', skiprows=1)
    
    n_features = train.shape[1] - 1
    
    X_train = train[:, :n_features]
    X_valid = valid[:, :n_features]
    X_test = test[:, :n_features]
    
    y_train = train[:, n_features:]
    y_valid = valid[:, n_features:]
    y_test = test[:, n_features:]
    
    print('Building the MLP...')

    # Build MLP
    model = Sequential()
    model.add(Dense(units=num_units, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dropout(rate=dropout_in))

    for k in range(n_hidden_layers):
        model.add(Dense(units=num_units//2))
        model.add(BatchNormalization(momentum=alpha, epsilon=epsilon))
        model.add(Activation('relu'))
        model.add(Dropout(rate=dropout_hidden))

    model.add(Dense(units=1))
    model.add(BatchNormalization(momentum=alpha, epsilon=epsilon))
    model.add(Activation('relu'))

    model.compile(loss='squared_hinge', optimizer=Adam(learning_rate=LR), metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1, validation_data=(X_valid, y_valid), callbacks=[es])

    test_results = model.evaluate(X_test, y_test, verbose=1)
    print('Loss: ' + str(test_results[0]))
    print('Accuracy: ' + str(test_results[1]))
    print('Error: ' + str(1-test_results[1]))
    
    with open(model_path, 'w') as f:
        f.write(model.to_json())

    model.save_weights(weights_path)