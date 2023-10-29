# *** train dataset configs ***
train_images_path = 'dataset/train/images'
train_labels_path = 'dataset/train/labels.csv'

# *** Uncomment if you have validation dataset ***
# val_images_path = 'dataset/validation/images'
# val_labels_path = 'dataset/validation/labels.csv'


# *** the full path is derived from save_dir and project: {save_dir}/{project} ***
save_dir = "runs"  # the directory which weights and logs are saved.
project = "exp"  # the project name

# *** Train Parameters ***
input_w = 1024
input_h = 128
batch_size = 16
downsample_factor = 4
epochs = 100
save_freq = 1000
patience = 30  # early stopping patience
lr = 0.002  # learning rate
decay = 1e-6  # sgd weight decay
momentum = 0.9  # sgd momentum
