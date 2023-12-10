# Integrating Vision Transformer-Based Self-Supervised Pre-training for Few-Shot Learning on Meta-Album

This study explores the use of visual transformers (ViT) in combina- tion with self-supervised learning (SSL) for few-shot learning (FSL) in the context of the Meta-Album dataset. Despite the rapid progress of deep learning, the need for large amounts of labeled data remains a challenge, often resulting in models prone to over- fitting due to data scarcity. This study aims to solve these problems by combining the feature extraction capabilities of SSL pre-trained visoin transformer and the efficiency of FSL in processing limited annotated data. Our method leverages pre-trained ViT in an SSL en- vironment and subsequently leverages the learned representations via FSL techniques. The experimental framework will compare two models: a standard FSL algorithms and a ViT-based model pre-trained using SSL. Our study showed that combining pre-train SSL ViT model with FSL techniques improved the performance of the model on few-shot classification tasks on the Meta-Album dataset, setting a new benchmark for Meta-Album on cross- domain and within-domain FSL. 


## Prerequisites
The code require a Python 3.0 or above to run

```
pip install -r requirements.txt
```

## Data

You can use the Download_Data.ipynb to download All Datasets and it will create a folder for them to store locally.

## Running Locally

Just run

## Running on Google Colab

To run the model on Google Colab, upload the zip file using the Colab folder upload feature and execute the following command at the beginning of the notebook:


Afterward, proceed to run the code cells in the notebook.

## Dataset Structure
Organize your dataset as follows:
```python
Data/
├── dataset1/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels.csv
│   │
│   └── info.json
├── dataset2/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels.csv
│   │
│   └── info.json
...
```
The labels.csv file should include columns for FILE_NAME, CATEGORY, and SUPER_CATEGORY, corresponding to the image file names and their labels.

## Usage
Execute all cells in the notebook to train the model and evaluate its accuracy.

## License
This project is released under the MIT License. Refer to the LICENSE.md file for more details.

## Acknowledgments
The model is built upon the Vision Transformer architecture, specifically using the vit-base-patch16-224-in21k model developed by Google Brain researchers and provided by the Hugging Face transformers library.

We extend our gratitude to the creators of the ViT and PyTorch libraries for making the development of this model possible.
