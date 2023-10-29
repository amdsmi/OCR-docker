import os
import argparse

import tensorflow as tf
from tensorflow import keras
import tf2onnx


def keras2onnx(weight, opset, output_name):
    model_ckpt2 = tf.keras.models.load_model(weight)

    model = keras.models.Model(model_ckpt2.get_layer(name='input_data').input,
                               [model_ckpt2.get_layer(name='tf.keras.backend.ctc_decode').output[0],
                                model_ckpt2.get_layer(name='dense2').output])

    tf2onnx.convert.from_keras(model, opset=opset, output_path=output_name)

    print("Model successfully converted to onnx.")


def parse_args():
    desc = "Convert easy-ocr model to onnx"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--model", type=str, help="path to the saved model dir")
    parser.add_argument("--opset", type=int, help="opset to use for onnx conversion", default=14)
    parser.add_argument("--output_name", type=str, default="prediction_model.onnx", help="converted model name")

    return parser.parse_args()


#
if __name__ == '__main__':

    args = parse_args()

    assert args.model is not None and os.path.isdir(args.model), "model does not exist."

    keras2onnx(args.model, args.opset, args.output_name)
