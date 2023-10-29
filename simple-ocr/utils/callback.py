import os.path
from pathlib import Path

import tensorflow as tf

import config.config as cfg
from utils.general import increment_path

__has_val_dataset = hasattr(cfg, "val_images_path") and hasattr(cfg, "val_labels_path")

if __has_val_dataset:
    early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_ctc_loss_accuracy', patience=cfg.patience)
else:
    early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='ctc_loss_accuracy', patience=cfg.patience)

__save_dir_path = increment_path(Path(cfg.save_dir) / cfg.project)

if __has_val_dataset:
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(__save_dir_path, "saved_models", "sm-{epoch:04d}"),
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        mode='max',
        monitor='val_ctc_loss_accuracy')
else:
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(__save_dir_path, "saved_models", "sm-{epoch:04d}"),
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        mode='max',
        monitor='ctc_loss_accuracy')

tensor_board_cb = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join(__save_dir_path, "tensorboard_logs"),
    histogram_freq=0,
    write_graph=True,
    write_images=True,
    update_freq="epoch",
    profile_batch=0,
    embeddings_freq=0,
    embeddings_metadata=None,
)

csv_logger_cb = tf.keras.callbacks.CSVLogger(os.path.join(__save_dir_path, "csv_log", "result.csv"), separator=",",
                                             append=False)
