from tensorflow.keras.layers import LSTM, Dense, Bidirectional

class LSTMModelGenerator():
    def create_lstm_layers(self, inputs, units=256, cell=LSTM, n_layers=2, dropout=0.3, bidirectional=False):

        previous_outputs = []
        for i in range(n_layers):
            if i == 0:
                # first layer
                if bidirectional:
                    previous_outputs.append(Bidirectional(cell(units,
                                     return_sequences=True, dropout=dropout))(inputs))
                else:
                    previous_outputs.append(cell(units,
                                     return_sequences=True, dropout=dropout)(inputs))
            elif i == n_layers - 1:
                # last layer
                if bidirectional:
                    previous_outputs.append(Bidirectional(cell(units,
                                    return_sequences=False, dropout=dropout))(previous_outputs[-1]))
                else:
                    previous_outputs.append(cell(units,
                                    return_sequences=False, dropout=dropout)(previous_outputs[-1]))
            else:
                # hidden layers
                if bidirectional:
                    previous_outputs.append(Bidirectional(cell(units,
                                    return_sequences=True, dropout=dropout))(previous_outputs[-1]))
                else:
                    previous_outputs.append(cell(units,
                                    return_sequences=True, dropout=dropout)(previous_outputs[-1]))
        
        previous_outputs.append(Dense(1)(previous_outputs[-1]))
        return previous_outputs[-1]
        # model = keras.Model(inputs=inputs, outputs=previous_outputs[-1], name="mnist_model")
        # model.compile(optimizer=optimizer, loss=loss, metrics=["mean_absolute_error"])