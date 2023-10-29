import argparse
import os
import cv2

from tqdm import tqdm
import pandas as pd
from predict import SimpleOCR

from difflib import SequenceMatcher


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def evaluate(model, dataset, label, characters, use_gpu):
    detector = SimpleOCR(model, characters, use_gpu=use_gpu)

    data = pd.read_csv(label)

    true_label = 0
    similarity = 0

    for image, label in tqdm(zip(data['image'], data['label'])):

        image = cv2.imread(os.path.join(dataset, image))

        result = detector.recognize(image)[0].strip(' ')

        if result == label:
            true_label += 1
            similarity += 1
        else:
            similarity += similar(result, label)

    return {'strict': true_label / len(data), 'similarity': similarity / len(data)}


def parse_args():
    desc = "Face Spoofing Detector Evaluate"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '--model',
        type=str,
        help='input ONNX model path')

    parser.add_argument(
        '--dataset',
        type=str,
        help='dataset path to evaluate')

    parser.add_argument(
        '--label',
        type=str,
        help='label path')

    parser.add_argument(
        '--chars',
        type=str,
        help='model characters path')

    parser.add_argument(
        '--use_gpu',
        default=False,
        action='store_true',
        help='use gpu for inference')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    assert args.model is not None and os.path.isfile(args.model), 'model does not exist.'
    assert args.dataset is not None and os.path.isdir(args.dataset), 'dataset does not exist.'
    assert args.label is not None and os.path.isfile(args.label), 'label does not exist.'
    assert args.chars is not None and os.path.isfile(args.chars), 'Characters does not exist.'

    result = evaluate(args.model, args.dataset, args.label, args.chars, args.use_gpu)
    strict = round(result['strict'], 3)
    similarity = round(result['similarity'], 3)
    print(f'the strict accuracy is: {strict}')
    print(f'the similarity accuracy is: {similarity}')
