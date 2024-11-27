# dataloader.py

import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset, Dataset
from torchvision import transforms, datasets
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def plot_class_distribution(dataloader, class_names, split_name):
    labels = []
    for _, label in dataloader:
        labels.extend(label.numpy())
    count = Counter(labels)
    classes = list(class_names)
    counts = [count[i] for i in range(len(classes))]

    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title(f'Class Distribution in {split_name} Set')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def print_class_distribution(dataloader, class_names, split_name):
    labels = []
    for _, label in dataloader:
        labels.extend(label.numpy())
    count = Counter(labels)
    print(f"Class Distribution in {split_name}:")
    for class_idx, num in count.items():
        print(f"  {class_names[class_idx]}: {num} samples")
    print("-" * 30)

def parse_args():
    parser = argparse.ArgumentParser(description="Cervical Cancer Dataset Loader")
    parser.add_argument('--dataset', type=str, choices=['Herlev', 'Mendeley', 'sipakmed'], required=True,
                        help='Name of the dataset to use')
    parser.add_argument('--classification', type=str, choices=['binary', 'multiclass'], default='binary',
                        help='Type of classification: binary or multiclass')
    parser.add_argument('--augmentation', type=int, choices=[0, 1, 2], default=0,
                        help='Level of data augmentation severity (0 = none, 3 = highest)')
    parser.add_argument('--root_dir', type=str, default='./datasets',
                        help='Path to the root directory containing datasets')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for DataLoader')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of training data to use for validation (between 0 and 1)')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Fraction of training data to use for testing (between 0 and 1)')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Visualize sample images from the dataset')
    return parser.parse_args()

def denormalize(images, mean, std):
    """
    Denormalize images for visualization.
    """
    mean = torch.tensor(mean).reshape(1, 3, 1, 1)
    std = torch.tensor(std).reshape(1, 3, 1, 1)
    images = images * std + mean
    return images

def visualize_images(dataloader, mean, std, class_names):
    images, labels = next(iter(dataloader))
    images = denormalize(images, mean, std)
    images = images.permute(0, 2, 3, 1).numpy()  # Convert to HWC format

    plt.figure(figsize=(12, 8))
    for i in range(min(8, len(images))):
        plt.subplot(2, 4, i + 1)
        plt.imshow(np.clip(images[i], 0, 1))
        plt.title(f"Label: {class_names[labels[i].item()]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def get_cancer_mappings(dataset_name):
    if dataset_name == 'Herlev':
        cancer_classes = [
            'carcinima_in_situ',
            'light_dysplastic',
            'moderate_dysplastic',
            'severe_dysplastic'
        ]
        non_cancer_classes = [
            'normal_columnar',
            'normal_intermediate',
            'normal_superficiel'
        ]
    elif dataset_name == 'Mendeley':
        cancer_classes = [
            'HSIL',
            'LSIL',
            'SCC'
        ]
        non_cancer_classes = [
            'NL'
        ]
    elif dataset_name == 'sipakmed':
        cancer_classes = [
            'im_Dyskeratotic',
            'im_Koilocytotic'
        ]
        non_cancer_classes = [
            'im_Superficial_Intermediate',
            'im_Parabasal',
            'im_Metaplastic'
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return cancer_classes, non_cancer_classes

def get_transforms(mean, std, augmentation_level):
    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    augmentation_transforms = []

    if augmentation_level >= 1:
        augmentation_transforms.extend([
            transforms.RandomHorizontalFlip(),
        ])

    if augmentation_level >= 2:
        augmentation_transforms.extend([
            transforms.RandomRotation(10),
        ])

    if augmentation_level >= 3:
        augmentation_transforms.extend([
        ])

    if augmentation_level >= 3:
        transform = transforms.Compose(augmentation_transforms + base_transforms)
    else:
        # Include initial Resize before other augmentations
        transform = transforms.Compose([transforms.Resize((224, 224))] + augmentation_transforms + base_transforms)

    return transform

def compute_mean_std(dataset):
    """
    Compute the mean and standard deviation of the dataset.
    """
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    mean = 0.
    std = 0.
    nb_samples = 0.

    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)
        nb_samples += images.size(0)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= nb_samples
    std /= nb_samples

    return mean.tolist(), std.tolist()

def get_data_loaders(args):
    data_dir = os.path.join(args.root_dir, args.dataset)
    # Temporary transform to convert images to tensors for mean/std computation
    temp_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load the dataset without normalization to compute mean and std
    dataset_for_stats = datasets.ImageFolder(root=data_dir, transform=temp_transform)
    class_to_idx = dataset_for_stats.class_to_idx
    print("Class to Index Mapping:", class_to_idx)

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    if args.classification == 'binary':
        cancer_classes, non_cancer_classes = get_cancer_mappings(args.dataset)
        cancer_classes = set(cancer_classes)
        non_cancer_classes = set(non_cancer_classes)

    # Compute mean and std
    print("Computing mean and standard deviation of the dataset...")
    mean, std = compute_mean_std(dataset_for_stats)
    print(f"Computed mean: {mean}")
    print(f"Computed std: {std}")

    # Define transforms for training, validation, and testing
    train_transform = get_transforms(mean, std, args.augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_transform = val_transform  # Typically, same as validation

    # Define the target_transform function for binary classification
    def binary_target_transform(class_idx):
        class_name = idx_to_class[class_idx]
        if class_name in cancer_classes:
            return 1  # Cancer
        else:
            return 0  # Non-cancer

    # Load the main dataset without transforms
    if args.classification == 'binary':
        dataset = datasets.ImageFolder(root=data_dir, transform=None, target_transform=binary_target_transform)
        num_classes = 2
        class_names = ['Non-Cancer', 'Cancer']
    else:
        dataset = datasets.ImageFolder(root=data_dir, transform=None)
        num_classes = len(dataset.classes)
        class_names = dataset.classes  # List of class names

    # Extract targets for stratification
    if args.classification == 'binary':
        targets = [binary_target_transform(label) for _, label in dataset.imgs]
    else:
        targets = [label for _, label in dataset.imgs]

    # Initialize StratifiedShuffleSplit for Test Set
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_split, random_state=42)
    train_val_indices, test_indices = next(sss.split(np.zeros(len(targets)), targets))

    # Now, split train_val_indices into Training and Validation using StratifiedShuffleSplit
    train_val_targets = [targets[i] for i in train_val_indices]
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=args.val_split / (1 - args.test_split), random_state=42)
    train_indices, val_indices = next(sss_val.split(np.zeros(len(train_val_indices)), train_val_targets))

    # Define Subsets using the same dataset instance to maintain consistent class_to_idx
    train_subset_original = Subset(dataset, [train_val_indices[i] for i in train_indices])
    val_subset_original = Subset(dataset, [train_val_indices[i] for i in val_indices])
    test_subset_original = Subset(dataset, test_indices)

    # Wrap subsets with their respective transforms
    train_subset = SubsetWithTransform(train_subset_original, transform=train_transform)
    val_subset = SubsetWithTransform(val_subset_original, transform=val_transform)
    test_subset = SubsetWithTransform(test_subset_original, transform=test_transform)

    # Handle class imbalance by computing class weights based on Training set
    if args.classification == 'binary':
        # Extract targets from training dataset
        train_targets = [binary_target_transform(dataset.imgs[i][1]) for i in train_subset_original.indices]
    else:
        train_targets = [dataset.imgs[i][1] for i in train_subset_original.indices]

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_targets), y=train_targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Create WeightedRandomSampler for Training set
    if args.classification in ['binary', 'multiclass']:
        sampler_weights = [class_weights[label] for label in train_targets]
        sampler = WeightedRandomSampler(sampler_weights, num_samples=len(sampler_weights), replacement=True)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Validation and Test loaders without sampling
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.visualize:
        visualize_images(train_loader, mean, std, class_names)
    # After getting class_names in dataloader.py
    print("Class Names:", class_names)

    return train_loader, val_loader, test_loader, num_classes, class_names

if __name__ == "__main__":
    args = parse_args()
    train_loader, val_loader, test_loader, num_classes, class_names = get_data_loaders(args)

    print_class_distribution(train_loader, class_names, "Training")
    print_class_distribution(val_loader, class_names, "Validation")
    print_class_distribution(test_loader, class_names, "Test")

    plot_class_distribution(train_loader, class_names, "Training")
    plot_class_distribution(val_loader, class_names, "Validation")
    plot_class_distribution(test_loader, class_names, "Test")
