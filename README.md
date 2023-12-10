# Integrating Vision Transformer-Based Self-Supervised Pre-training for Few-Shot Learning on Meta-Album

This study explores the use of visual transformers (ViT) in combina- tion with self-supervised learning (SSL) for few-shot learning (FSL) in the context of the Meta-Album dataset. Despite the rapid progress of deep learning, the need for large amounts of labeled data remains a challenge, often resulting in models prone to over- fitting due to data scarcity. This study aims to solve these problems by combining the feature extraction capabilities of SSL pre-trained visoin transformer and the efficiency of FSL in processing limited annotated data. Our method leverages pre-trained ViT in an SSL en- vironment and subsequently leverages the learned representations via FSL techniques. The experimental framework will compare two models: a standard FSL algorithms and a ViT-based model pre-trained using SSL. Our study showed that combining pre-train SSL ViT model with FSL techniques improved the performance of the model on few-shot classification tasks on the Meta-Album dataset, setting a new benchmark for Meta-Album on cross- domain and within-domain FSL. 


## Prerequisites
The code require a Python 3.0 or above to run.
Run the following command to install required libraries:
```
pip install -r requirements.txt
```

## Data (Meta-Album) 

For the dataset please refer to the original paper to download: https://meta-album.github.io

Note: You should have all 30 datasets (mini version) downloaded in a folder called "Datasets" then place this folder inside 'dino_fsl' folder.

## Dataset Structure
Organize your dataset as follows:
```python
Datasets/
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
Before running the code, please cd to downloaded dino_fsl folder:
```
cd /path/to/dino_fsl/folder
```

### For within domain experiments run the following commands:

#### Matching Networks (vitb16):
```
python3 -m Code.Within_Domain.main "MatchingNetworks" "facebook/dino-vitb16" "/path/to/Code/folder"

```
#### Matching Networks (vits16):
```
python3 -m Code.Within_Domain.main "MatchingNetworks" "facebook/dino-vits16" "/path/to/Code/folder"

```
#### Prototypical Networks (vitb16):
```
python3 -m Code.Within_Domain.main "PrototypicalNetworks" "facebook/dino-vitb16" "/path/to/Code/folder"

```

#### Prototypical Networks (vits16):
```
python3 -m Code.Within_Domain.main "PrototypicalNetworks" "facebook/dino-vits16" "/path/to/Code/folder"

```
### For Cross domain experiments run the following commands:

#### Matching Networks (vitb16):
```
python3 -m Code.Cross_Domain.main "MatchingNetworks" "facebook/dino-vitb16" "/path/to/Code/folder"

```
#### Matching Networks (vits16):
```
python3 -m Code.Cross_Domain.main "MatchingNetworks" "facebook/dino-vits16" "/path/to/Code/folder"

```
#### Prototypical Networks (vitb16):
```
python3 -m Code.Cross_Domain.main "PrototypicalNetworks" "facebook/dino-vitb16" "/path/to/Code/folder"

```

#### Prototypical Networks (vits16):
```
python3 -m Code.Cross_Domain.main "PrototypicalNetworks" "facebook/dino-vits16" "/path/to/Code/folder"



## License
This project is released under the MIT License. Refer to the LICENSE.md file for more details.

## Acknowledgments
This project has been greatly enhanced by the utilization of external resources and the contributions of the broader open-source community. We would like to extend our sincere gratitude to the following:

Sicara's Easy Few-Shot Learning: A significant portion of the code used in this project was adapted from Sicara's "Easy Few-Shot Learning" repository on GitHub. Their comprehensive and well-structured implementation provided an excellent foundation for our work in few-shot learning. You can find their repository here: [Easy Few-Shot Learning](https://github.com/sicara/easy-few-shot-learning/tree/master) by Sicara.

Facebook's DINO-ViT Models: Our project leverages the Facebook/dino-vit models, which have been instrumental in advancing our understanding and capabilities in the field of visual transformers. The models' robustness and versatility have significantly contributed to the success of our project.
We are immensely grateful to these contributors for their pioneering work and for making their resources available to the community, thereby enabling us and others to build upon their innovations and push the boundaries of machine learning and artificial intelligence.
