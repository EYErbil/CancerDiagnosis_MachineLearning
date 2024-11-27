# train.py

import os
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import get_model
from dataloader import get_data_loaders
import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0.0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path to save the model checkpoint.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, val_acc, model):

        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_acc = val_acc
            self.save_checkpoint(val_loss, val_acc, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.best_loss = val_loss
            self.best_acc = val_acc
            self.save_checkpoint(val_loss, val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_acc, model):
        '''Saves model when validation loss decreases.'''
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Models')
    parser.add_argument('--model_name', type=str, required=True, choices=['model1', 'model2', 'model3','model4','model5','model6','model7'],
                        help='Name of the model to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model weights')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for Adam optimizer (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for Adam optimizer (default: 1e-5)')
    parser.add_argument('--dataset', type=str, choices=['Herlev', 'Mendeley', 'sipakmed'], required=True,
                        help='Name of the dataset to use')
    parser.add_argument('--classification', type=str, choices=['binary', 'multiclass'], default='binary',
                        help='Type of classification: binary or multiclass')
    parser.add_argument('--augmentation', type=int, choices=[0, 1, 2, 3], default=2,
                        help='Level of data augmentation severity (0 = none, 3 = highest)')
    parser.add_argument('--root_dir', type=str, default='./datasets', help='Root directory of the dataset')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of training data to use for validation (between 0 and 1)')
    # Added --test_split argument
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Fraction of training data to use for testing (between 0 and 1)')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience (default: 30)')
    parser.add_argument('--delta', type=float, default=0.0, help='Minimum change in validation loss to qualify as improvement (default: 0.0)')
    parser.add_argument('--visualize', action='store_true', help='Visualize sample images from the dataset')
    return parser.parse_args()

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, model_name, save_dir):
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Acc', color='blue')
    plt.plot(epochs, val_accuracies, label='Validation Acc', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_metrics.png'))
    plt.close()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, model_name, save_dir, device, patience, delta):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=delta,
                                   path=os.path.join(save_dir, f'{model_name}_earlystop.pth'))

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs} for {model_name.upper()}')

        # Training phase
        model.train()
        running_loss = 0.0
        corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())
        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Validation phase
        model.eval()
        running_loss = 0.0
        corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                corrects += torch.sum(preds == labels)

        epoch_loss_val = running_loss / len(val_loader.dataset)
        epoch_acc_val = corrects.double() / len(val_loader.dataset)
        val_losses.append(epoch_loss_val)
        val_accuracies.append(epoch_acc_val.item())
        print(f'Validation Loss: {epoch_loss_val:.4f} Acc: {epoch_acc_val:.4f}')

        # Adjust the learning rate based on validation loss
        scheduler.step(epoch_loss_val)

        early_stopping(epoch_loss_val, epoch_acc_val, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the best model weights
    model.load_state_dict(torch.load(early_stopping.path))
    final_checkpoint_path = os.path.join(save_dir, f'{model_name}_best.pth')
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f'Best model saved at {final_checkpoint_path}')

    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, model_name, save_dir)

    return model

if __name__ == "__main__":
    args = parse_arguments()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data loaders and number of classes
    train_loader, val_loader, test_loader, num_classes, class_names = get_data_loaders(args)

    # Get the model
    model = get_model(args.model_name, num_classes=num_classes)
    model = model.to(device)

    # Define the loss function and optimizer
    # For class imbalance, use weighted loss
    if args.classification in ['binary', 'multiclass']:
        # Compute class weights from training set
        # Already handled in dataloader.py via WeightedRandomSampler
        # Here, use standard CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Initialize the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Use ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                     patience=5, verbose=True)

    # Train the model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        model_name=args.model_name,
        save_dir=args.save_dir,
        device=device,
        patience=args.patience,
        delta=args.delta
    )
