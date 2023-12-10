import sys
from torchvision import transforms
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter, RandomResizedCrop
from .data_preparation import DomainDataset
from .experiment import startExperiment
from transformers import ViTFeatureExtractor, ViTForImageClassification
import numpy as np
import os
import random


if len(sys.argv) > 3:
    project_folder_path = sys.argv[3]
    type_of_network = sys.argv[1]
    modelName = sys.argv[2]
    print("type_of_network: ",type_of_network)
else:
    print("No argument provided, please follow the readme file")
    sys.exit(1)  # Exit the script if no argument is provided


# Change the current working directory
os.chdir(project_folder_path)


# Load the feature extractor from the pre-trained ViT model (transfer-learning)
feature_extractor = ViTFeatureExtractor.from_pretrained(modelName)

# Use the feature extractor's normalization parameters
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to 224x224
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),  # Normalize the images
])

domains_names =['AWA','BRD','BTS','CRS','DOG','FLW','FNG','INS','BCT','ACT_40','ACT_410','APL',
                'INS_2','MD_5_BIS','MD_6','MD_MIX','MED_LF','PLK','PLT_DOC','PLT_NET','PLT_VIL',
                'PNU','PRT','RESISC','RSD','RSICB','SPT','TEX','TEX_ALOT','TEX_DTD']

domain_results = []#one_shot_results,five_shots_results,ten_shots_results,twenty_shots_results

full_experiment_results = {}

embedding_dim = 2042
N = 5 #
num_tasks = 30
num_epochs = 1


# Path to data
data_folder = '../Datasets/'

# Define a file name for the results
results_filename = f"../Results/{type_of_network}_Within_Domain_experiment_results_{modelName}.txt"



print('datasets: ', len(domains_names))
with open(results_filename, "w") as file:
    for domain in domains_names:
        print('domain: ', domain)
        domain_path = os.path.join(data_folder, domain)
        print('domain_path: ', domain_path)
        domain_results = startExperiment(DomainDataset(domain_path, transform), modelName, domain, embedding_dim, N, num_tasks, num_epochs, type_of_network)
        full_experiment_results[domain] = domain_results

        file.write(f"Domain: {domain}\n")
        for shot in [1, 5, 10, 20]:
            shot_key = shot
            mean_accuracy = domain_results[shot_key]["mean"]
            std_dev = domain_results[shot_key]["std"]
            conf_interval = domain_results[shot_key]["ci"]
            file.write(f"5-way {shot}-shot: {num_tasks} Tasks Mean Accuracy: {mean_accuracy:.2f}%, Std Dev: {std_dev:.2f}%, 95% CI: {conf_interval:.2f}%\n")
        file.write("---------------------------------------------------------\n")

    file.write("##################################################\n")

    # Calculate the mean accuracy for all domains for each shot
    accuracies_per_shot = {1: [], 5: [], 10: [], 20: []}
    mean_stds_per_shot = {1: [], 5: [], 10: [], 20: []}  # Store mean STDs for each shot
    mean_cis_per_shot = {1: [], 5: [], 10: [], 20: []}  # Store mean CIs for each shot

    for domain, results in full_experiment_results.items():
        for shot in mean_stds_per_shot.keys():
            shot_key = shot
            mean_stds_per_shot[shot].append(results[shot_key]["std"])  # Append STD for each domain
            mean_cis_per_shot[shot].append(results[shot_key]["ci"])  # Append CI for each domain

    for shot in mean_stds_per_shot.keys():
        mean_accuracy = np.mean([result[shot]["mean"] for result in full_experiment_results.values()])
        mean_std = np.mean(mean_stds_per_shot[shot])  # Calculate mean of STDs
        mean_ci = np.mean(mean_cis_per_shot[shot])  # Calculate mean of CIs
        txt = f"Mean accuracy for {shot}-shot across all domains: {mean_accuracy:.2f}%, Mean Std Dev: {mean_std:.2f}%, Mean 95% CI: {mean_ci:.2f}%\n"
        print(txt)
        file.write(txt)
