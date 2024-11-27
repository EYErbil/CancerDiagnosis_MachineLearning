# test_ensemble.py

import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    multilabel_confusion_matrix
)
import matplotlib.pyplot as plt
import numpy as np
from dataloader import get_data_loaders
from models import get_model
import itertools
from PIL import Image

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate Ensemble of Trained Models')
    parser.add_argument('--models', nargs='+', required=True, choices=[
        'model1', 'model2', 'model3', 'model4', 'model5', 'model6', 'model7'],
                        help='List of model names to evaluate')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                        help='List of checkpoint paths corresponding to the models')
    parser.add_argument('--ensemble_method', type=str, required=True, choices=[
        'max_prob', 'avg_prob', 'majority_vote'],
                        help='Ensemble method to use: max_prob, avg_prob, or majority_vote')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for DataLoader (default: 32)')
    parser.add_argument('--dataset', type=str, choices=['Herlev', 'Mendeley', 'sipakmed'], required=True,
                        help='Name of the dataset to use')
    parser.add_argument('--classification', type=str, choices=['binary', 'multiclass'], default='binary',
                        help='Type of classification: binary or multiclass (default: binary)')
    parser.add_argument('--augmentation', type=int, choices=[0, 1, 2, 3], default=0,
                        help='Level of data augmentation severity (0 = none, 3 = highest)')
    parser.add_argument('--root_dir', type=str, default='./datasets',
                        help='Root directory of the dataset (default: ./datasets)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of training data to use for validation (between 0 and 1, default: 0.2)')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Fraction of training data to use for testing (between 0 and 1, default: 0.1)')
    parser.add_argument('--num_visualize', type=int, default=10,
                        help='Number of test images to visualize (default: 10)')
    parser.add_argument('--save_dir', type=str, default='./evaluation_results_ensemble',
                        help='Directory to save evaluation results and visualizations (default: ./evaluation_results_ensemble)')
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
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    if classification_type == 'binary':
        # Define target_transform for binary classification
        class_to_idx = dataset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        cancer_classes, non_cancer_classes = get_cancer_mappings(dataset_name)
        cancer_classes = set(cancer_classes)
        non_cancer_classes = set(non_cancer_classes)

        def binary_target_transform(label):
            class_name = idx_to_class[label]
            if class_name in cancer_classes:
                return 1  # Cancer
            else:
                return 0  # Non-cancer

        targets = [binary_target_transform(label) for _, label in dataset.imgs]
    else:
        # Multiclass classification
        targets = [label for _, label in dataset.imgs]

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

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
            'carcinoma_in_situ',
            'light_dysplastic',
            'moderate_dysplastic',
            'severe_dysplastic'
        ]
        non_cancer_classes = [
            'normal_columnar',
            'normal_intermediate',
            'normal_superficial'
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

def ensemble_evaluate(models, test_loader, criterion, device, class_names, classification_type, save_dir, num_visualize, mean, std, ensemble_method):
    """
    Evaluate an ensemble of models on the test dataset.

    Args:
        models (list): List of PyTorch models.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the evaluation on.
        class_names (list): List of class names.
        classification_type (str): 'binary' or 'multiclass'.
        save_dir (str): Directory to save evaluation results.
        num_visualize (int): Number of test images to visualize.
        mean (list): Mean used for normalization.
        std (list): Standard deviation used for normalization.
        ensemble_method (str): Ensemble method ('max_prob', 'avg_prob', 'majority_vote').

    Returns:
        None
    """
    for model in models:
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

            # Get outputs from all models
            outputs_list = [model(inputs) for model in models]

            # Ensemble methods
            if ensemble_method == 'max_prob':
                # Take the maximum probability across models for each class
                softmax_outputs = [nn.functional.softmax(outputs, dim=1) for outputs in outputs_list]
                stacked_softmax = torch.stack(softmax_outputs)  # Shape: (num_models, batch_size, num_classes)
                max_probs, max_indices = torch.max(stacked_softmax, dim=0)
                preds = torch.argmax(max_probs, dim=1)
                ensemble_probs = max_probs
            elif ensemble_method == 'avg_prob':
                # Average the probabilities across models
                avg_outputs = torch.mean(torch.stack(outputs_list), dim=0)
                ensemble_probs = nn.functional.softmax(avg_outputs, dim=1)
                preds = torch.argmax(ensemble_probs, dim=1)
            elif ensemble_method == 'majority_vote':
                # Majority vote on predicted classes
                preds_list = [torch.argmax(outputs, dim=1) for outputs in outputs_list]
                preds_stack = torch.stack(preds_list, dim=1)  # Shape: (batch_size, num_models)
                preds, _ = torch.mode(preds_stack, dim=1)
                # For probabilities, take the average of softmax outputs
                softmax_outputs = [nn.functional.softmax(outputs, dim=1) for outputs in outputs_list]
                ensemble_probs = torch.mean(torch.stack(softmax_outputs), dim=0)
            else:
                raise ValueError(f"Unsupported ensemble_method: {ensemble_method}")

            # Compute loss using ensemble probabilities if possible, else use the first model's outputs
            if ensemble_method in ['max_prob', 'avg_prob']:
                loss = criterion(ensemble_probs, labels)
            else:  # For majority_vote, using cross-entropy loss on predictions is not straightforward
                # Here, we compute loss using the average probabilities
                loss = criterion(ensemble_probs, labels)

            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # For ROC and PR curves
            if classification_type == 'binary':
                if ensemble_method in ['max_prob', 'avg_prob']:
                    probs = ensemble_probs[:, 1]  # Probability of class 1
                else:
                    # For majority_vote, take the average probability for class 1
                    probs = ensemble_probs[:, 1]
                all_probs.extend(probs.cpu().numpy())
            elif classification_type == 'multiclass':
                probs = ensemble_probs
                all_probs.extend(probs.cpu().numpy())

            # Store images for visualization
            if len(all_images) < num_visualize:
                all_images.extend(inputs.cpu())
                all_true_labels.extend(labels.cpu().numpy())
                all_pred_labels.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = corrects.double() / len(test_loader.dataset)
    print(f'\nEnsemble Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

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

    # ROC Curves for Multiclass Classification
    elif classification_type == 'multiclass':
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = len(class_names)
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(
                [1 if label == i else 0 for label in all_labels],
                [prob[i] for prob in all_probs]
            )
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curves for all classes
        plt.figure()
        colors = plt.cm.get_cmap('tab10').colors
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)],
                     lw=2, label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) - Multiclass')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_dir, 'roc_curve_multiclass.png'))
        plt.close()
        print(f'ROC curves for multiclass saved to {os.path.join(save_dir, "roc_curve_multiclass.png")}')

        # Precision-Recall Curves for Multiclass Classification
        precision = dict()
        recall = dict()
        pr_auc = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(
                [1 if label == i else 0 for label in all_labels],
                [prob[i] for prob in all_probs]
            )
            pr_auc[i] = auc(recall[i], precision[i])

        # Plot Precision-Recall curves for all classes
        plt.figure()
        for i in range(n_classes):
            plt.plot(recall[i], precision[i], color=colors[i % len(colors)],
                     lw=2, label=f'PR curve of class {class_names[i]} (area = {pr_auc[i]:.2f})')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Multiclass')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(save_dir, 'precision_recall_curve_multiclass.png'))
        plt.close()
        print(f'Precision-Recall curves for multiclass saved to {os.path.join(save_dir, "precision_recall_curve_multiclass.png")}')

    # Save Classification Report
    save_classification_report(report, save_dir)

    # Visualize Sample Predictions
    visualize_sample_predictions(
        all_images,
        all_true_labels,
        all_pred_labels,
        class_names,
        classification_type,
        save_dir,
        num_visualize,
        mean,
        std
    )

def plot_confusion_matrix(cm, class_names, classification_type, save_dir, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plot and save the confusion matrix.

    Args:
        cm (np.ndarray): Confusion matrix.
        class_names (list): List of class names.
        classification_type (str): 'binary' or 'multiclass'.
        save_dir (str): Directory to save the plot.
        normalize (bool): Whether to normalize the confusion matrix.
        title (str): Title of the plot.
        cmap: Colormap for the plot.

    Returns:
        None
    """
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        cm_display = cm_normalized
    else:
        print('Confusion matrix, without normalization')
        cm_display = cm

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_display, interpolation='nearest', cmap=cmap)
    plt.title(f'{title} ({classification_type.capitalize()})')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm_display.max() / 2.
    for i, j in itertools.product(range(cm_display.shape[0]), range(cm_display.shape[1])):
        plt.text(j, i, format(cm_display[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_display[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    cm_filename = f'confusion_matrix_{classification_type}.png'
    plt.savefig(os.path.join(save_dir, cm_filename))
    plt.close()
    print(f'Confusion matrix saved to {os.path.join(save_dir, cm_filename)}')

def plot_roc_curve(fpr, tpr, roc_auc, save_dir):
    """
    Plot and save the ROC curve.

    Args:
        fpr (dict or list): False positive rates.
        tpr (dict or list): True positive rates.
        roc_auc (dict or float): Area under the curve.
        save_dir (str): Directory to save the plot.

    Returns:
        None
    """
    plt.figure()
    if isinstance(fpr, dict):
        # Multiclass ROC
        for i in range(len(fpr)):
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    else:
        # Binary ROC
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if isinstance(fpr, dict):
        plt.title('Receiver Operating Characteristic (ROC) - Multiclass' if len(fpr) > 2 else 'ROC')
    else:
        plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    roc_filename = 'roc_curve_multiclass.png' if isinstance(fpr, dict) else 'roc_curve.png'
    plt.savefig(os.path.join(save_dir, roc_filename))
    plt.close()
    print(f'ROC curve saved to {os.path.join(save_dir, roc_filename)}')

def plot_precision_recall_curve(recall, precision, pr_auc, save_dir):
    """
    Plot and save the Precision-Recall curve.

    Args:
        recall (dict or list): Recall values.
        precision (dict or list): Precision values.
        pr_auc (dict or float): Area under the curve.
        save_dir (str): Directory to save the plot.

    Returns:
        None
    """
    plt.figure()
    if isinstance(recall, dict):
        # Multiclass Precision-Recall
        for i in range(len(recall)):
            plt.plot(recall[i], precision[i], lw=2, label=f'Class {i} (AUC = {pr_auc[i]:.2f})')
    else:
        # Binary Precision-Recall
        plt.plot(recall, precision, color='blue',
                 lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if isinstance(recall, dict):
        plt.title('Precision-Recall Curve - Multiclass' if len(recall) > 2 else 'Precision-Recall Curve')
    else:
        plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    pr_filename = 'precision_recall_curve_multiclass.png' if isinstance(recall, dict) else 'precision_recall_curve.png'
    plt.savefig(os.path.join(save_dir, pr_filename))
    plt.close()
    print(f'Precision-Recall curve saved to {os.path.join(save_dir, pr_filename)}')

def save_classification_report(report, save_dir):
    """
    Save the classification report to a text file.

    Args:
        report (str): Classification report.
        save_dir (str): Directory to save the report.

    Returns:
        None
    """
    report_path = os.path.join(save_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f'Classification report saved to {report_path}')

def visualize_sample_predictions(images, true_labels, pred_labels, class_names, classification_type, save_dir, num_visualize, mean, std):
    """
    Visualize a grid of sample predictions.

    Args:
        images (list): List of image tensors.
        true_labels (list): List of true labels.
        pred_labels (list): List of predicted labels.
        class_names (list): List of class names.
        classification_type (str): 'binary' or 'multiclass'.
        save_dir (str): Directory to save the visualization.
        num_visualize (int): Number of images to visualize.
        mean (list): Mean used for normalization.
        std (list): Standard deviation used for normalization.

    Returns:
        None
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

    if len(args.models) != len(args.checkpoints):
        raise ValueError("The number of models and checkpoints must be the same.")

    # Create directory to save evaluation results
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Compute mean and std independently
    data_dir = os.path.join(args.root_dir, args.dataset)
    mean, std, class_names_computed = compute_mean_std(
        data_dir, args.classification, args.dataset)
    print(f"Computed Mean: {mean}")
    print(f"Computed Std: {std}\n")

    # Get data loaders
    # Note: We're not using the mean and std from dataloader.py
    _, _, test_loader, num_classes, class_names = get_data_loaders(args)

    # Initialize the models
    models = []
    for model_name, checkpoint_path in zip(args.models, args.checkpoints):
        model = get_model(model_name, num_classes=num_classes)
        model = model.to(device)

        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint '{checkpoint_path}' for model '{model_name}'...")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"Checkpoint for model '{model_name}' loaded successfully.\n")
        else:
            print(f"No checkpoint found at '{checkpoint_path}'. Exiting.")
            return

        models.append(model)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate the ensemble of models
    ensemble_evaluate(
        models=models,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        classification_type=args.classification,
        save_dir=args.save_dir,
        num_visualize=args.num_visualize,
        mean=mean,
        std=std,
        ensemble_method=args.ensemble_method
    )

if __name__ == "__main__":
    main()
