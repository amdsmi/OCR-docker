import cv2
import numpy
import numpy as np
import onnxruntime


class SimpleOCR:
    def __init__(self, model_path: str, characters_path: str, use_gpu=False):

        if use_gpu:
            self.model = onnxruntime.InferenceSession(model_path,
                                                      providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        else:
            self.model = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        input_cfg = self.model.get_inputs()[0]
        self.model_input_name = input_cfg.name
        self.model_input_size = input_cfg.shape[1:3]
        self.out_net_names = [self.model.get_outputs()[0].name]

        with open(characters_path, "r") as f:
            self.characters = list(f.read().strip())

        self.char_to_labels = {char: idx for idx, char in enumerate(self.characters)}
        self.labels_to_char = {val: key for key, val in self.char_to_labels.items()}
        self._warm_up()

    def recognize(self, image: numpy.ndarray):
        return self.recognize_batch([image])[0]

    def recognize_batch(self, images):
        blob = [self._preprocess(image, self.model_input_size[0], self.model_input_size[1]) for image in
                images]
        preds = self.model.run(self.out_net_names, {self.model_input_name: blob})
        decoded_predictions = self._decode_batch_predictions(preds[0])
        results = [res.strip() for res in decoded_predictions]

        return results

    def _decode_batch_predictions(self, pred):
        output_text = []
        for res in pred:
            result_str = ''
            for c in res:
                if len(self.characters) > c >= 0:
                    result_str += self.labels_to_char[c]
            output_text.append(result_str)

        return output_text

    def _warm_up(self):
        np.random.seed(seed=1234)
        input_data = np.random.randint(255, size=(self.model_input_size[0], self.model_input_size[1], 3),
                                       dtype=np.uint8)
        self.recognize(input_data)

    def _preprocess(self, image, w, h):
        image = self._pad_if_need(image, w, h)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = (image / 255.).astype(np.float32)
        image = image.T
        image = np.expand_dims(image, axis=-1)
        return image

    def _pad_if_need(self, image, width, height):
        h, w, _ = image.shape

        if w == width and h == height:
            return image

        target_height = height
        target_width = width

        if w >= width and h <= height:

            width_ratio = target_width / float(w)
            new_height = int(width_ratio * h)

            resized_image = cv2.resize(image, (target_width, new_height))
            final_image = cv2.copyMakeBorder(resized_image, 0, target_height - new_height, 0, 0, cv2.BORDER_CONSTANT,
                                             None, value=(255, 255, 255))

            return self._pad_if_need(final_image, width, height)

        else:

            height_ratio = target_height / float(h)
            new_width = int(height_ratio * w)

            resized_image = cv2.resize(image, (new_width, target_height))

            if new_width >= width:
                return self._pad_if_need(resized_image, width, height)

            final_image = cv2.copyMakeBorder(resized_image, 0, 0, target_width - new_width, 0, cv2.BORDER_CONSTANT,
                                             None, value=(255, 255, 255))
            return self._pad_if_need(final_image, width, height)
