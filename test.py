# test.py

import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
from dataloader import get_data_loaders
from models import get_model
import itertools
from PIL import Image

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate Trained Models')
    parser.add_argument('--model_name', type=str, required=True, choices=['model1', 'model2', 'model3','model4','model5','model6','model7'],
                        help='Name of the model to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for DataLoader')
    parser.add_argument('--dataset', type=str, choices=['Herlev', 'Mendeley', 'sipakmed'], required=True,
                        help='Name of the dataset to use')
    parser.add_argument('--classification', type=str, choices=['binary', 'multiclass'], default='binary',
                        help='Type of classification: binary or multiclass')
    parser.add_argument('--augmentation', type=int, choices=[0, 1, 2, 3], default=0,
                        help='Level of data augmentation severity (0 = none, 3 = highest)')
    parser.add_argument('--root_dir', type=str, default='./datasets', help='Root directory of the dataset')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of training data to use for validation (between 0 and 1)')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Fraction of training data to use for testing (between 0 and 1)')
    parser.add_argument('--num_visualize', type=int, default=10,
                        help='Number of test images to visualize (default: 10)')
    parser.add_argument('--save_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results and visualizations')
    # Added visualize argument
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Flag to visualize images during data loading')
    return parser.parse_args()

def compute_mean_std(data_dir, classification_type, dataset_name):
    """
    Compute the mean and standard deviation of the dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        classification_type (str): 'binary' or 'multiclass'.
        dataset_name (str): Name of the dataset ('Herlev', 'Mendeley', 'sipakmed').

    Returns:
        mean (list): Mean values for each channel.
        std (list): Standard deviation for each channel.
        class_names (list): List of class names.
    """
    # Define the transform to convert images to tensors
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load the dataset without normalization
    if classification_type == 'binary':
        # Define target_transform for binary classification
        def binary_target_transform(class_idx, class_to_idx, cancer_classes):
            class_name = idx_to_class[class_idx]
            if class_name in cancer_classes:
                return 1  # Cancer
            else:
                return 0  # Non-cancer

        # Get cancer and non-cancer classes
        cancer_classes, non_cancer_classes = get_cancer_mappings(dataset_name)
        cancer_classes = set(cancer_classes)
        non_cancer_classes = set(non_cancer_classes)

        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        class_to_idx = dataset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        # Compute targets
        targets = [binary_target_transform(label, class_to_idx, cancer_classes) for _, label in dataset.imgs]
    else:
        # Multiclass classification
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        targets = [label for _, label in dataset.imgs]

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    mean = 0.0
    std = 0.0
    total_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples

    return mean.tolist(), std.tolist(), dataset.classes

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

def evaluate_model(model, test_loader, criterion, device, class_names, classification_type, save_dir, num_visualize, mean, std):
    model.eval()
    running_loss = 0.0
    corrects = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_images = []
    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # For binary classification ROC
            if classification_type == 'binary':
                probs = nn.functional.softmax(outputs, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
            elif classification_type == 'multiclass':
                probs = nn.functional.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())

            # Store images for visualization
            if len(all_images) < num_visualize:
                all_images.extend(inputs.cpu())
                all_true_labels.extend(labels.cpu().numpy())
                all_pred_labels.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = corrects.double() / len(test_loader.dataset)
    print(f'\nTest Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("Classification Report:")
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, classification_type, save_dir)

    # ROC Curve and AUC for Binary Classification
    if classification_type == 'binary':
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        plot_roc_curve(fpr, tpr, roc_auc, save_dir)
        # Precision-Recall Curve
        precision, recall, pr_thresholds = precision_recall_curve(all_labels, all_probs)
        pr_auc = auc(recall, precision)
        plot_precision_recall_curve(recall, precision, pr_auc, save_dir)

    # Save Classification Report
    save_classification_report(report, save_dir)

    # Visualize Sample Predictions
    visualize_sample_predictions(all_images, all_true_labels, all_pred_labels, class_names, classification_type, save_dir, num_visualize, mean, std)

def plot_confusion_matrix(cm, class_names, classification_type, save_dir, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f'{title} ({classification_type.capitalize()})')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_{classification_type}.png'))
    plt.close()
    print(f'Confusion matrix saved to {os.path.join(save_dir, f"confusion_matrix_{classification_type}.png")}')

def plot_roc_curve(fpr, tpr, roc_auc, save_dir):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()
    print(f'ROC curve saved to {os.path.join(save_dir, "roc_curve.png")}')

def plot_precision_recall_curve(recall, precision, pr_auc, save_dir):
    plt.figure()
    plt.plot(recall, precision, color='blue',
             lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(save_dir, 'precision_recall_curve.png'))
    plt.close()
    print(f'Precision-Recall curve saved to {os.path.join(save_dir, "precision_recall_curve.png")}')

def save_classification_report(report, save_dir):
    report_path = os.path.join(save_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f'Classification report saved to {report_path}')

def visualize_sample_predictions(images, true_labels, pred_labels, class_names, classification_type, save_dir, num_visualize, mean, std):
    """
    Visualize a grid of sample predictions.
    """
    num_images = min(num_visualize, len(images))
    cols = min(num_images, 5)
    rows = (num_images + cols - 1) // cols
    plt.figure(figsize=(cols * 4, rows * 4))
    for idx in range(num_images):
        image = images[idx]
        true_label = class_names[true_labels[idx]]
        pred_label = class_names[pred_labels[idx]]

        # Denormalize the image
        image = denormalize_image(image, mean, std)

        plt.subplot(rows, cols, idx + 1)
        plt.imshow(image)
        plt.title(f'True: {true_label}\nPred: {pred_label}', fontsize=12)
        plt.axis('off')
    plt.tight_layout()
    visualization_path = os.path.join(save_dir, 'sample_predictions.png')
    plt.savefig(visualization_path)
    plt.close()
    print(f'Sample predictions saved to {visualization_path}')

def denormalize_image(tensor, mean, std):
    """
    Denormalize a tensor image using mean and std.

    Args:
        tensor (torch.Tensor): Tensor image of size (C, H, W).
        mean (list): Mean values for each channel.
        std (list): Standard deviation for each channel.

    Returns:
        np.ndarray: Denormalized image in HWC format.
    """
    image = tensor.numpy().transpose((1, 2, 0))  # Convert to HWC
    mean = np.array(mean)
    std = np.array(std)
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return image

def main():
    args = parse_arguments()

    # Create directory to save evaluation results
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Compute mean and std independently
    data_dir = os.path.join(args.root_dir, args.dataset)
    mean, std, class_names_computed = compute_mean_std(data_dir, args.classification, args.dataset)
    print(f"Computed Mean: {mean}")
    print(f"Computed Std: {std}\n")

    # Get data loaders
    # Note: We're not using the mean and std from dataloader.py
    train_loader, val_loader, test_loader, num_classes, class_names = get_data_loaders(args)

    # Initialize the model
    model = get_model(args.model_name, num_classes=num_classes)
    model = model.to(device)

    # Load the trained model checkpoint
    if os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint '{args.checkpoint}'...")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print("Checkpoint loaded successfully.\n")
    else:
        print(f"No checkpoint found at '{args.checkpoint}'. Exiting.")
        return

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        classification_type=args.classification,
        save_dir=args.save_dir,
        num_visualize=args.num_visualize,
        mean=mean,
        std=std
    )

if __name__ == "__main__":
    main()
