import pandas as pd
import os


class DatasetParser(object):
    '''
    Class to parse the dataset and split it into train and test sets.
    '''

    def __init__(self, images_dir, labels_path):
        """
        Initializes the dataset parser.
        :param images_dir: The path to the images.
        """
        self.images_dir = images_dir
        self.characters = set()
        self.data = pd.read_csv(labels_path)
        self._load_dataset()
        self.char_to_labels = {char: idx for idx, char in enumerate(self.characters)}
        self.labels_to_char = {val: key for key, val in self.char_to_labels.items()}
        self.random_seed = 1234

    def _load_dataset(self):
        """
        Loads the dataset2 from the given path.
        :return: A list of images.
        """
        concatenated_labels = ''.join(self.data["label"].to_numpy())

        self.characters = set(list(concatenated_labels))
        self.characters = sorted(self.characters)
        self.max_len = self.data.label.str.len().max()
        self.data["image"] = os.path.join(os.path.normpath(self.images_dir), "") + self.data["image"].astype(str)
