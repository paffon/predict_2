from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def nearest_power_of_two(num: int) -> int:
    """
    Find the power of two nearest to the given number from above.

    :param num: The input number.

    :return: The nearest power of two.
    """
    power = 1

    while power < num:
        power *= 2

    return power


def build_model(input_len: int):
    # Define and return your neural network architecture
    # using TensorFlow/Keras
    layers_sizes = [nearest_power_of_two(2 * input_len) for _ in range(4)]

    sequence = [layers.InputLayer(input_shape=(input_len,))]

    for size in layers_sizes:
        sequence.append(layers.Dense(size, activation='relu'))

    sequence.append(layers.Dense(1, activation='sigmoid'))

    model = tf.keras.Sequential(sequence)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model


class MyNeuralNetwork:
    def __init__(self, input_len: int):
        # Define your neural network architecture here
        self.model = build_model(input_len)

    def train(self, x_df: pd.DataFrame, y: pd.Series):
        x_train, x_eval, y_train, y_eval = train_test_split(
            x_df, np.array(y),
            test_size=0.2,
            random_state=42
        )

        # Train your model
        self.model.fit(
            x_train, y_train,
            epochs=500, validation_data=(x_eval, y_eval)
        )

    def predict(self, x_test):
        # Make predictions using the trained model
        predictions = self.model.predict(x_test)
        # Convert predictions to binary labels
        binary_predictions = np.where(predictions > 0.5, 1, -1)
        return binary_predictions
