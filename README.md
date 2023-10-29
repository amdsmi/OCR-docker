## Simple OCR Docekr

This is the project that build the docker image in order to train and test simple OCR model.
The Docker file can be build using script below:
```
$ docker build -t ocr_docker:0.1
```

After building the docker image it can be used in purposes of training and the testing OCR models.
First, whole dataset should be structured as below graph is shown. 

```
 dataset_____
            |______train
            |          |______ images
            |          |______ labels.csv
            |
            |______validation
            |                |______ images
            |                |______ labels.csv
            |
            |______test
                       |______ images
                       |______ labels.csv

```

If there is a costume config.py file should be put in dataset directory.
In addition, dataset directory, another directory should be created named 'output' in order to retrieve training and test result

The docker image can be run using script below

```
docker run -it -v <absolute dataset path>:/app/dataset -v <absolute output directory path>:/app/output <image_name>
```

After training and testing the model the result can be seen in the output directory

An example of config file is in the repository that can control model hyperparameter and dataset path 
