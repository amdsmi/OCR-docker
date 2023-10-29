import argparse
import os
from pathlib import Path

import tensorflow as tf
from config import config as cfg
from utils import datagenerator as dg
from utils import dataparser as dp
from utils import callback as cb
from utils.general import increment_path
from model.model import build_model


def train(args):
    # check for correct path
    assert os.path.isdir(cfg.train_images_path), "Training images does not exist!"
    assert os.path.isfile(cfg.train_labels_path), "Training labels does not exist!"

    # Load data
    train_data_parser = dp.DatasetParser(cfg.train_images_path, cfg.train_labels_path)

    save_dir_path = increment_path(Path(cfg.save_dir) / cfg.project, mkdir=True)

    if not os.path.isdir(os.path.join(save_dir_path, "char_list")):
        os.mkdir(os.path.join(save_dir_path, "char_list"))

    with open(os.path.join(save_dir_path, "char_list", "characters.txt"), "w") as f:
        f.write(''.join(train_data_parser.characters))

    os.mkdir(os.path.join(save_dir_path, "csv_log"))

    train_data_generator = dg.DataGenerator(images_path=train_data_parser.data['image'].to_numpy(),
                                            labels=train_data_parser.data['label'].to_numpy(),
                                            characters=train_data_parser.characters,
                                            batch_size=args.batch_size,
                                            img_width=cfg.input_w,
                                            img_height=cfg.input_h,
                                            downsample_factor=cfg.downsample_factor,
                                            max_length=train_data_parser.max_len,
                                            shuffle=True
                                            )

    has_val_dataset = hasattr(cfg, "val_images_path") and hasattr(cfg, "val_labels_path")

    if has_val_dataset:
        assert os.path.isdir(cfg.val_images_path), "Validation images does not exist!"
        assert os.path.isfile(cfg.val_labels_path), "Validation labels does not exist!"

        val_data_parser = dp.DatasetParser(cfg.val_images_path, cfg.val_labels_path)

        validation_data_generator = dg.DataGenerator(images_path=val_data_parser.data['image'].to_numpy(),
                                                     labels=val_data_parser.data['label'].to_numpy(),
                                                     characters=train_data_parser.characters,
                                                     batch_size=args.batch_size,
                                                     img_width=cfg.input_w,
                                                     img_height=cfg.input_h,
                                                     downsample_factor=cfg.downsample_factor,
                                                     max_length=train_data_parser.max_len,
                                                     shuffle=False
                                                     )

    if args.checkpoint is not None and os.path.isdir(args.checkpoint):
        print('Loading pre-trained model weights from {}'.format(args.checkpoint))
        model = tf.keras.models.load_model(args.checkpoint)
    else:
        model = build_model(cfg.input_w, cfg.input_h, len(train_data_parser.characters), train_data_parser.max_len)

    model.summary()

    model.fit(train_data_generator,
              validation_data=validation_data_generator if has_val_dataset else None,
              epochs=args.epochs,
              callbacks=[cb.early_stop_cb, cb.cp_callback, cb.tensor_board_cb, cb.csv_logger_cb]
              )


def parse_args():
    parser = argparse.ArgumentParser("text recognition by Simple OCR")

    parser.add_argument('--epochs', type=int, default=cfg.epochs, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=cfg.batch_size, help='Batch size for training')
    parser.add_argument('--checkpoint', type=str, help='Load a previously trained model')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    train(args)
