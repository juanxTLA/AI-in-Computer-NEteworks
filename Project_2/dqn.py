import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LayerNormalization, LSTM
from keras.optimizers import Adam

class DQN(tf.Module):
    def __init__(self, learning_rate=0.01, state_size=4,
                 action_size=2, hidden_layer_sizes=[10], step_size=1,
                 name='QNetwork'):

        super(DQN, self).__init__(name=name)

        # Define the model using the Sequential API
        self.model = Sequential()

        self.model.add(LSTM(10, input_shape=(step_size, state_size)))

        # Build hidden layers
        for i, hidden_size in enumerate(hidden_layer_sizes):
            self.model.add(Dense(hidden_size, activation='relu', name=f'h{i + 1}'))
            self.model.add(LayerNormalization())

        # Fully connected layer for output
        self.model.add(Dense(action_size, activation='linear', name='output'))

        # Compile the model
        self.model.compile(optimizer=Adam(learning_rate),
                           loss='mean_squared_error')
        # print("done")

    def train(self, states, epochs, targets, batchSize):
        # Train the model
        self.model.fit(states, targets, epochs=epochs, batch_size=batchSize, verbose=0)
        # print("Training done. . .")

    def predict(self, states):
        # Predict Q-values for a batch of states
        return self.model.predict(states, verbose=0)