import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

# Input layer with shape (28, 28, 1)
inputs = layers.Input(shape=(28, 28, 1))
x = inputs

x = layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(units=128, activation='relu')(x)
x = layers.Dense(units=10, activation='softmax')(x)

# Build model
model = tf.keras.Model(inputs=inputs, outputs=x)
# Compile model with Adam optimizer and categorical_crossentropy loss
model.compile(loss='categorical_crossentropy', optimizer=Adam())
