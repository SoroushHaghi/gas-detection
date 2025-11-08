import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D, GRU, Dense
from tensorflow.keras.models import Model

def build_cnn_model(input_shape, num_classes):
    """Builds a 1D-CNN model for time-series classification."""
    input_layer = Input(shape=input_shape)
    # Block 1
    x = Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # Block 2
    x = Conv1D(filters=64, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # Output
    x = GlobalAveragePooling1D()(x)
    output_layer = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=input_layer, outputs=output_layer, name="1D_CNN")

def build_gru_model(input_shape, num_classes):
    """Builds a GRU model for time-series classification."""
    input_layer = Input(shape=input_shape)
    x = GRU(units=32, return_sequences=True)(input_layer)
    x = GRU(units=32, return_sequences=False)(x)
    output_layer = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=input_layer, outputs=output_layer, name="GRU")

def build_mlp_model(input_shape, num_classes):
    """Builds a simple MLP model for 2D statistical data."""
    input_layer = Input(shape=input_shape)
    # Hidden Layer 1
    x = Dense(128, activation='relu')(input_layer)
    x = tf.keras.layers.Dropout(0.3)(x)
    # Hidden Layer 2
    x = Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    # Output Layer
    output_layer = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer, name="MLP")