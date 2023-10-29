# Recognition Model

## Abstract

This project used [Simple-OCR](https://github.com/agiagents/simple-ocr) version 1.0.1 to recognize iranian license plate number.

## Config

```
input_w = Input_w
input_h = Input_h
batch_size = Batch_size
downSample_factor = Downsample_factor
epochs = Epochs
save_freq = Save_freq
patience = Patience 
lr = Lr  
decay = Decay
momentum = Momentum
```

## Dataset description

The dataset information is shown below:


###Train: train_sample_number images with below statistics
###Validation: val_sample_number images with below statistics
###Test: test_sample_number images with below statistics


## Result

### Train and validation

|          Loss           |          Accuracy          |
|:-----------------------:|:--------------------------:|
| ![loss](loss_plot_path) | ![accuracy](acc_plot_path) |

| Metric   |  Train     | Validation   |
|----------|------------|--------------|
| Loss     | train_loss | val_loss     |
| Accuracy | train_acc  | val_acc      |

### Evaluation

| Metric             | Test     | 
|--------------------|----------| 
| restrict_Accuracy  | test_res |
| similaritty_Accuracy | test_sim |
