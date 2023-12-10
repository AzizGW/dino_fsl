import torch
from torch.utils.data import DataLoader
from .data_preparation import create_N_way_few_shot_dataset, split_dataset
from ..model import DinoViTEmbedding
from ..matching_networks import MatchingNetworks
from ..training import train
from ..testing import testTheModel
import gc
import numpy as np

device = None
# Check if CUDA is available
  # Use "cuda" for NVIDIA GPUs or "cpu" if neither applies. 
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Print CUDA device count
else:
    device = torch.device("cpu")  
    print("No CUDA devices available")
# For Apple M1, M2 users, please use 'mps' for device to utilize the gpu

def startExperiment(dataset, embedding_model_name,domain,embedding_dim,N,num_tasks,num_epochs):

    # Stores results for different shot scenarios
    one_shot_results, five_shots_results, ten_shots_results, twenty_shots_results = [], [], [], []

    # Stores results for different shot scenarios
    results_dict = {
        1: {"mean": None, "std": None, "ci": None},
        5: {"mean": None, "std": None, "ci": None},
        10: {"mean": None, "std": None, "ci": None},
        20: {"mean": None, "std": None, "ci": None}
    }
    # Iterating over different shot scenarios
    for shots in [1,5,10,20]:
        print('Shots:', shots)
        all_task_accuracies = []  # Store all task accuracies for current shot

        train_dataset, test_dataset = split_dataset(dataset)
        print("Train dataset size:", len(train_dataset))
        print("Test dataset size:", len(test_dataset))

        # Extract labels and count unique classes
        train_labels = [label for _, label in train_dataset]
        test_labels = [label for _, label in test_dataset]

        num_classes_train = len(set(train_labels))
        num_classes_test = len(set(test_labels))

        print("Number of classes in train dataset:", num_classes_train)
        print("Number of classes in test dataset:", num_classes_test)


        global_best_training_loss = 10000000.00
        global_best_model_state = None

        # Init model
        dino_vit_embedding = DinoViTEmbedding(embedding_model_name, embedding_dim).to(device)
        matching_networks = MatchingNetworks(backbone=dino_vit_embedding,feature_dimension=embedding_dim).to(device)


        for task in range(num_tasks):
            print("Training Task:", task)

            # Create N-way few-shot datasets
            train_support_set, train_query_set = create_N_way_few_shot_dataset(train_dataset, shots, N)

            # Data loaders for support, query, and validation sets
            train_support_loader = DataLoader(train_support_set, batch_size=64, shuffle=True, num_workers=0)
            train_query_loader = DataLoader(train_query_set, batch_size=64, shuffle=True, num_workers=0)

            # Train the model
            best_model_state,best_training_loss = train(matching_networks, train_support_loader, train_query_loader, num_epochs,device)

            # Update the global best model state if the current task's model is better
            if best_training_loss < global_best_training_loss:
                global_best_training_loss = best_training_loss
                global_best_model_state = best_model_state

            # clear cache
            #torch.cuda.empty_cache()
            del train_support_loader
            del train_query_loader
            del train_support_set
            del train_query_set
            gc.collect()

        # Save the state dictionary
        torch.save(global_best_model_state, 'best_model_state.pth')
        model_state = torch.load('best_model_state.pth')
        matching_networks.load_state_dict(model_state)

        # Tesing episodes
        for task in range(num_tasks):
            print("Testing Task:", task)

            # Create N-way few-shot datasets
            test_support_set,test_query_set = create_N_way_few_shot_dataset(test_dataset, shots, N)
            test_support_loader = DataLoader(test_support_set, batch_size=64, shuffle=True, num_workers=0)
            test_query_loader = DataLoader(test_query_set, batch_size=64, shuffle=True, num_workers=0)

            # Test the model
            task_accuracy = testTheModel(test_support_loader,test_query_loader, matching_networks,device)
            all_task_accuracies.append(task_accuracy*100)
            # clear cache
            #torch.cuda.empty_cache()
            
        # Calculate mean accuracy and 95% confidence interval for the tasks
            del test_support_loader
            del test_query_loader
            del test_support_set
            del test_query_set
            gc.collect()
        mean_accuracy = np.mean(all_task_accuracies)
        std_accuracy = np.std(all_task_accuracies)
        confidence_interval = 1.96 * std_accuracy / np.sqrt(num_tasks)


        # Store results
        shot_key = shots
        results_dict[shot_key]["mean"] = mean_accuracy
        results_dict[shot_key]["std"] = std_accuracy
        results_dict[shot_key]["ci"] = confidence_interval


        print(f"Domain: {domain}. 5-way {shots}-shot learning with {num_tasks} training tasks and {num_epochs} epochs for each task: Mean accuracy: {mean_accuracy:.2f}%, Standard Deviation: {std_accuracy:.2f}%, 95% CI: {confidence_interval:.2f}%")
        print("---------------------------------------------------------")


    return results_dict