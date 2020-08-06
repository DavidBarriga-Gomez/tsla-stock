class LSTMProcessedData:
    def __init__(self, x_train, y_train, x_test, y_test, n_features, scaler):
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test 
        self.n_features=n_features
        self.scaler=scaler
        
    