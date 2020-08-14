class LSTMProcessedData:
    def __init__(self, data_set, x_train, y_train, x_test, y_test, n_features, column_scalers, last_sequence):
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test 
        self.n_features=n_features
        self.column_scalers=column_scalers
        self.data_set=data_set
        self.last_sequence=last_sequence
        
    