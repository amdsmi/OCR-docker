import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

import config.config as cfg

print("Tensorflow version: ", tf.__version__)


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, input_length, label_length):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # On validation time, just return the computed loss
        return loss


def decode_output(pred):
    pred = pred[:, 2:]
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred,
                                       input_length=input_len,
                                       greedy=True)[0][0]

    return results


def build_model(img_width, img_height, num_class, max_char_size):
    # Inputs to the model
    input_img = layers.Input(shape=(img_width, img_height, 1), name='input_data', dtype='float32')

    labels = layers.Input(name='input_label', shape=[max_char_size], dtype='float32')
    input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')

    # inputs = keras.layers.Input(shape = (img_width, img_height, 1),name="image")
    conv1 = keras.layers.Conv2D(32, (5, 5), activation='relu', padding='SAME', name='Conv1')(input_img)
    normal1 = layers.BatchNormalization(name='batch_norm1')(conv1)
    max1 = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(normal1)
    conv2 = keras.layers.Conv2D(64, (5, 5), activation='relu', padding='SAME', name='Conv2')(max1)
    normal2 = layers.BatchNormalization(name='batch_norm2')(conv2)
    max2 = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(normal2)
    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='SAME', name='Conv3')(max2)
    normal3 = layers.BatchNormalization(name='batch_norm3')(conv3)
    max3 = keras.layers.MaxPooling2D(pool_size=(1, 2), name="pool3")(normal3)
    conv4 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='SAME', name='Conv4')(max3)
    normal4 = layers.BatchNormalization(name='batch_norm4')(conv4)
    max4 = keras.layers.MaxPooling2D(pool_size=(1, 2), name="pool4")(normal4)
    conv5 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='SAME', name='Conv5')(max4)
    normal5 = layers.BatchNormalization(name='batch_norm5')(conv5)
    # We have used two max pool with pool size and strides of 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 512. Reshape accordingly before
    # passing it to RNNs

    new_shape = ((img_width // 4), (img_height // 16) * 256)
    x = layers.Reshape(target_shape=new_shape, name='reshape')(normal5)
    x = layers.Dense(64, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2, name='lstm1'), name='bidirectional1')(
        x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25, name='lstm2'),
                             name='bidirectional2')(x)

    # Predictions
    softmax_out = layers.Dense(num_class + 1, activation='softmax', name='dense2')(x)

    # decode output
    output_labels = keras.backend.ctc_decode(softmax_out, tf.multiply(keras.backend.ones(tf.shape(softmax_out)[0]),
                                                                      softmax_out.shape[1]), greedy=True)[0][0]

    # Calculate CTC
    ctc_out = CTCLayer(name='ctc_loss')(labels, softmax_out, input_length, label_length)

    # Define the model
    model = keras.models.Model(inputs=[input_img, labels, input_length, label_length],
                               outputs=[ctc_out, softmax_out, output_labels],
                               name='simple_ocr')

    # Optimizer
    sgd = keras.optimizers.SGD(learning_rate=cfg.lr,
                               decay=cfg.decay,
                               momentum=cfg.momentum,
                               nesterov=True,
                               clipnorm=5)

    # Compile the model and return
    model.compile(optimizer=sgd, metrics={"ctc_loss": "accuracy"})

    return model
