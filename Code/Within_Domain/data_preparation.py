import os
import csv
from PIL import Image
from torch.utils.data import Dataset
import random
import numpy as np

# Define the dataset class
class DomainDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.data_info = self._load_data_info()
        self.label_list = sorted(list(set([item['label'] for item in self.data_info])))

    def _load_data_info(self):
        data_info = []
        print('data_folder:',self.data_folder)
        labels_path = os.path.join(self.data_folder, 'labels.csv')
        print('labels_path:',labels_path)
        print(os.getcwd())
        try:
            with open(labels_path, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    image_path = os.path.join(self.data_folder, 'images', row['FILE_NAME'])
                    data_info.append({'image_path': image_path, 'label': row['CATEGORY']})
        except FileNotFoundError:
            print(f"No labels.csv file found in {self.data_folder}.")

        return data_info

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
          item = self.data_info[idx]
          image = Image.open(item['image_path']).convert('RGB')

          # Apply transformation
          if self.transform:
              image = self.transform(image)

          label = self.label_list.index(item['label'])
          return image, label



def create_N_way_few_shot_dataset(dataset, k, N):
    """
    Create N-way k-shot datasets for training and testing, ensuring that the classes
    in training and testing sets are distinct and mapped to new indices.

    Args:
        dataset: The dataset to split.
        k: Number of samples per class in the support set.
        N: Number of classes in each task.

    Returns:
        support_set, query_set
    """

    # Get the unique labels in the dataset
    unique_labels = list({label for _, label in dataset})
    total_classes = len(unique_labels)


    # Randomly sample N classes for the task
    classes = set(np.random.choice(unique_labels, N, replace=False))

    reverse = None
    def create_task_sets(classes_set):
        # Create label mapping for the selected classes
        label_mapping = {label: idx for idx, label in enumerate(classes_set)}
        reverse_mapping = {idx: label for label, idx in label_mapping.items()}

        task_data = {cls: [] for cls in classes_set}
        for image, label in dataset:
            if label in classes_set:
                task_data[label].append(image)

        support_set, query_set = [], []
        for cls in classes_set:
            np.random.shuffle(task_data[cls])
            support_samples = task_data[cls][:k]  # k-shot
            query_samples = task_data[cls][k:]  # Take all remaining samples after the first k
            # Apply label mapping
            support_set.extend([(image, label_mapping[cls]) for image in support_samples])
            query_set.extend([(image, label_mapping[cls]) for image in query_samples])
        
            
        random.shuffle(support_set)
        random.shuffle(query_set)

            # Reverse mapping and print labels
        final_support_labels = [reverse_mapping[label] for _, label in support_set]
        final_query_labels = [reverse_mapping[label] for _, label in query_set]
        #print(f"Final support set labels: {final_support_labels}, Size: {len(support_set)}")
        #print(f"Final query set labels: {final_query_labels}, Size: {len(query_set)}")



        return support_set, query_set

    # Create training and testing sets with mapped labels
    support_set, query_set = create_task_sets(classes)
    

    return support_set, query_set

def split_dataset(dataset, train_ratio=0.70):
    """
    Split the dataset into training and testing datasets.

    Args:
        dataset: The original dataset to split.
        train_ratio: The proportion of the dataset to include in the train split.

    Returns:
        train_dataset, test_dataset
    """

    unique_labels = list({label for _, label in dataset})
    random.shuffle(unique_labels)

    num_train = int(train_ratio * len(unique_labels))
    train_classes = set(unique_labels[:num_train])
    test_classes = set(unique_labels[num_train:])

    train_dataset = [item for item in dataset if item[1] in train_classes]
    test_dataset = [item for item in dataset if item[1] in test_classes]

    return train_dataset, test_dataset
