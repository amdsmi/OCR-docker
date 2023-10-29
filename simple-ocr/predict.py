import os
import argparse

import cv2

from simple_ocr import SimpleOCR


def parse_args():
    desc = "Simple-OCR inference"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', type=str, help='path to the onnx model')
    parser.add_argument('--source', nargs='+', type=str, help='image or list of images to read', required=True)
    parser.add_argument('--chars', type=str, help='model characters path')
    parser.add_argument('--use-gpu', action='store_true', default=False, help='use gpu for inference')

    return parser.parse_args()


def main(model: str, images: list, characters_path: str):

    image_list = []
    for image in images:
        image_list.append(cv2.imread(image))

    detector = SimpleOCR(model, characters_path, use_gpu=args.use_gpu)

    detector.recognize(cv2.imread(images[0]))

    results = detector.recognize_batch(image_list)

    for result in results:
        print(result)


if __name__ == '__main__':
    args = parse_args()

    assert args.model is not None and os.path.isfile(args.model), 'ONNX Model does not exist.'
    assert args.chars is not None and os.path.isfile(args.chars), 'Characters file does not exist.'

    main(args.model, args.source, args.chars)
