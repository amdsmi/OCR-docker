# Simple OCR

A simple implementation based on keras to detect texts and numbers. 

## Install Prerequisites

Install the required packages:

```
pip install -r requirements.txt
```

# Prepare Dataset

1. Put your training images under the `dataset/train/images` folder or specify your custom images path in the [config](config/config.py).
2. Put your training labels csv file under the `dataset/train/` folder or specify your custom labels path in the [config](config/config.py).

The dataset structure is shown below:

```shell
├──dataset/
  ├──train/
    ├──images/
      ├──image_1.jpg
      ├──image_2.jpg
      ├──image_3.jpg
      ...
      
    ├──labels.csv
```

The labels.csv file structure is shown below:

```shell
image,label
image_1.jpg,label_1
image_2.jpg,label_2
image_3.jpg,label_3
...
```

Validation dataset has the same structure as train:

```shell
├──dataset/
  ├──validation/
    ├──images/
      ├──image_1.jpg
      ├──image_2.jpg
      ├──image_3.jpg
      ...
      
    ├──labels.csv
```

Or your can specify your validation images path and labels path in the [config](config/config.py).

# Train
Specify your configurations in the [config](config/config.py) file and run:

```shell
python3 train.py
```

# Evaluation

```shell
python3 evaluate.py --model "onnx model path" --dataset "dataset_path" --label "label_path" --chars "model characters path"
```

# Inference

```shell
python3 predict.py --model "onnx model path" --source "images path" --chars "model characters path"
```

# Convert To ONNX

```shell
python3 convert_to_onnx.py --model "saved model path"
```

# References

[SimpleHTR](https://github.com/githubharald/SimpleHTR)
