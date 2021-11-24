if __name__ == "__main__":
    from tensorflow.keras.models import model_from_json
    from tensorflow.keras.optimizers import Adam
    import numpy as np

    # Load model and weights
    model_path = 'mlp_model4.json'
    weights_path = 'mlp_weights4.h5'

    with open(model_path, 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(weights_path)

    # Load test data
    test = np.loadtxt('test4.csv', delimiter=',', skiprows=1)

    n_features = test.shape[1] - 1

    X_test = test[:, :n_features]
    y_test = test[:, n_features:]
    
    # Evaluate model on test data
    LR = .001
    model.compile(loss='squared_hinge', optimizer=Adam(learning_rate=LR), metrics=['accuracy'])
    test_results = model.evaluate(X_test, y_test, verbose=1)
    print('Loss: ' + str(test_results[0]))
    print('Accuracy: ' + str(test_results[1]))
    print('Error: ' + str(1-test_results[1]))