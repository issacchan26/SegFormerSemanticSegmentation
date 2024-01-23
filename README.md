# Semantic Segmentation with Hugging Face SegFormer, PyTorch and Segments.ai
This repo provides no trainer version of Hugging Face SegFormer model in PyTorch framework. The dataset is built with Segments.ai and released to Hugging Face.

## Getting Started
This repo is tested with Conda environment and Python 3.9 under Linux os, please run below command to install dependencies
```
pip install -r requirements.txt
```

## Data Annotation
This repo is using [Segments.ai](https://segments.ai/) to annotate the images, please use [release_dataset.py](release_dataset.py) to release Segments.ai dataset to Hugging Face.
Before annotating the images, you may use [convert_color.py](convert_color.py) to convert RGB images into Grayscale images if needed.

## Transfer Learning
[train_hf.py](train_hf.py) provides Trainer version for fine-tuning the Hugging Face model and save the fine-tuned model in local.
[train.py](train.py) provides no Trainer version for fine-tuning the Hugging Face model with image augmentations to further improve model performance.  

## Model Training
Please use [train.py](train.py) to train the model, modify the below arguments before training
```
args = Params(
        hf_dataset_identifier = "issacchan26/gray_bullet",  # path to hugging face dataset
        pretrained_model_name = '/path to pretrained model folder from Hugging Face',  # path to pretrained model
        epochs = 100,
        lr = 0.0005,
        batch_size = 1,
        checkpoints_path = "/path to/checkpoints/"  # path to checkpoints saving folder
        )
```
## Inference
1. [test.py](test.py)  
  It is used to infer the validation dataset and provide comparison images between ground truth and prediction  
2. [infer_hf_ds.py](infer_hf_ds.py)  
  It is used to infer the dataset from Hugging Face  

Please modify the below path before running  
```
hf_dataset_identifier = "issacchan26/gray_bullet",  # path to your hugging face dataset
pretrained_model_name = '/path to/checkpoints/best',  # path to model folder
prediction_save_path = '/path to/prediction/', # path to saving folder
```
