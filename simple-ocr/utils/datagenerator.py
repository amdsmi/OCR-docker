import cv2
import numpy as np
from tensorflow import keras


def preprocess(image, w, h, resized=False):
    """
    :param image: cv2 image
    :param resized(bool): if True, resize image to w x h
    :param w: width of resized image
    :param h: height of resized image
    :return: preprocessed image
    """
    image = _pad_if_need(image, w, h)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if resized:
        image = cv2.resize(image, (w, h))
    image = (image / 255.).astype(np.float32)
    image = image.T
    image = np.expand_dims(image, axis=-1)

    return image


def _pad_if_need(image, width, height):
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

        return _pad_if_need(final_image, width, height)

    else:

        height_ratio = target_height / float(h)
        new_width = int(height_ratio * w)

        resized_image = cv2.resize(image, (new_width, target_height))

        if new_width >= width:
            return _pad_if_need(resized_image, width, height)

        final_image = cv2.copyMakeBorder(resized_image, 0, 0, target_width - new_width, 0, cv2.BORDER_CONSTANT,
                                         None, value=(255, 255, 255))
        return _pad_if_need(final_image, width, height)


class DataGenerator(keras.utils.Sequence):
    """Generates batches from a given dataset2.

    Args:
        df_dataset(pandas): training or validation pandas dataframe dataset consist of ['images] and ['labels']
        labels(list): all the corresponding labels
        char_map(dict): dictionary mapping char to labels
        batch_size(int): size of a single batch
        img_width(int): width of the resized
        img_height(int): height of the resized
        downsample_factor: by what factor did the CNN downsample the images
        max_length: maximum length of any code
        shuffle(bool): whether to shuffle data or not after each epoch
    Returns:
        batch_inputs: a dictionary containing batch inputs
        batch_labels: a batch of corresponding labels
    """

    def __init__(self,
                 images_path,
                 labels,
                 characters,
                 img_width,
                 img_height,
                 batch_size=8,
                 downsample_factor=4,
                 max_length=10,
                 shuffle=True
                 ):
        self.images_path = images_path
        self.labels = labels
        self.characters = np.array(characters)
        self.char_map = {char: idx for idx, char in enumerate(characters)}
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.downsample_factor = downsample_factor
        self.max_length = max_length
        self.shuffle = shuffle
        self.indices = np.arange(len(images_path))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.images_path) / self.batch_size))

    def __getitem__(self, idx):
        # 1. Get the next batch indices
        curr_batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        # 2. This isn't necessary but it can help us save some memory
        # as not all batches the last batch may not have elements
        # equal to the batch_size
        batch_len = len(curr_batch_idx)
        # 3. Instantiate batch arrays
        batch_images = np.ones((batch_len, self.img_width, self.img_height, 1), dtype=np.float32)
        batch_labels = np.ones((batch_len, self.max_length), dtype=np.float32)
        input_length = np.ones((batch_len, 1), dtype=np.int64) * (self.img_width // self.downsample_factor - 2)
        label_length = np.zeros((batch_len, 1), dtype=np.int64)

        for j, idx in enumerate(curr_batch_idx):
            preprocessed_img = preprocess(cv2.imread(self.images_path[idx]), self.img_width, self.img_height, False)
            # 3. Get the correpsonding label
            label = self.labels[idx].strip()

            label_map = [self.char_map[ch] for ch in label] + [-1 for _ in range(self.max_length - len(label))]

            batch_images[j] = preprocessed_img
            batch_labels[j] = label_map
            label_length[j] = len(label)

        batch_inputs = {
            'input_data': batch_images,
            'input_label': batch_labels,
            'input_length': input_length,
            'label_length': label_length,
        }
        return batch_inputs, np.zeros(batch_len).astype(np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
