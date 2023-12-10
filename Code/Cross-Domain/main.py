from torchvision import transforms
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter, RandomResizedCrop
from .data_preparation import DomainDataset
from .experiment import startExperiment
from transformers import ViTFeatureExtractor, ViTForImageClassification
import numpy as np
import os
import random

#-------------Step 1--------------------------------------
# Replace this with the absolute path of your "Code" folder
code_folder_path = '/path/to/Code'

# Change the current working directory
os.chdir(code_folder_path)
#-------------Step 2--------------------------------------

# Choose one of the following models and comment the other one.
modelName = 'facebook/dino-vitb16' #B16
# modelName = 'facebook/dino-vits16' #S16
#------------------------------------------------------------

# Load the feature extractor from the pre-trained ViT model (transfer-learning)
feature_extractor = ViTFeatureExtractor.from_pretrained(modelName)

# Use the feature extractor's normalization parameters
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to 224x224
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),  # Normalize the images
])



# List of all domain names
domains_names = ['AWA', 'BRD', 'BTS', 'CRS', 'DOG', 'FLW', 'FNG', 'INS',
               'BCT', 'ACT_40', 'ACT_410', 'APL', 'INS_2', 'MD_5_BIS',
               'MD_6', 'MD_MIX', 'MED_LF', 'PLK', 'PLT_DOC', 'PLT_NET',
               'PLT_VIL', 'PNU', 'PRT', 'RESISC', 'RSD', 'RSICB', 'SPT',
               'TEX', 'TEX_ALOT', 'TEX_DTD']


# Randomly split domains into training and testing sets
random.shuffle(domains_names)
num_train_domains = int(len(domains_names) * 0.7)  # For example, 70% for training
train_domains = domains_names[:num_train_domains]
test_domains = domains_names[num_train_domains:]

print('Train domain: ',train_domains)
print('Test domain: ',test_domains)

embedding_dim = 2042
N = 5  # Number of classes per task
num_tasks = 30
num_epochs = 1

# Path to data
data_folder = '../Datasets/'

# Define a file name for the results
results_filename = f"../Results/Cross_Domain_experiment_results_{modelName}.txt"


print('Starting Cross-Domain Experiment...')

results = startExperiment(transform,data_folder,train_domains,test_domains, modelName,embedding_dim,N,num_tasks,num_epochs)

# Conduct the experiment on the combined dataset
with open(results_filename, "w") as file:
    # Write the results for the combined dataset
    file.write("Cross-Domain Experiment Results:\n")
    for shot in [1, 5, 10, 20]:
        mean_accuracy = results[shot]["mean"]
        std_dev = results[shot]["std"]
        conf_interval = results[shot]["ci"]
        file.write(f"5-way {shot}-shot: {num_tasks} Tasks Mean Accuracy: {mean_accuracy:.2f}%, Std Dev: {std_dev:.2f}%, 95% CI: {conf_interval:.2f}%\n")
    file.write("---------------------------------------------------------\n")

