# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.regnet import RegNet_X_400MF_Weights,RegNet_X_800MF_Weights

from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_V2_S_Weights
)
from torchvision.models import vit_b_16, ViT_B_16_Weights


class Model1(nn.Module):
    """
    Model 1: Simple ANN with fully connected layers (no CNNs).
    """

    def __init__(self, input_size=224 * 224 * 3, num_classes=None):
        super(Model1, self).__init__()
        if num_classes is None:
            raise ValueError("num_classes must be specified and cannot be None.")
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Flatten the input tensor
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x  # Output shape: [batch_size, num_classes]


class Model2(nn.Module):
    """
    Model 2: CNN similar to Snoutnet.
    """

    def __init__(self, num_classes=None):
        super(Model2, self).__init__()
        if num_classes is None:
            raise ValueError("num_classes must be specified and cannot be None.")
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)  # Adjust the dimensions accordingly
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Output: [batch_size, 64, 56, 56]
        x = self.pool(F.relu(self.conv2(x)))  # Output: [batch_size, 128, 14, 14]
        x = self.pool(F.relu(self.conv3(x)))  # Output: [batch_size, 256, 3, 3]
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x  # Output shape: [batch_size, num_classes]


class Model3(nn.Module):
    """
    Model 3: Advanced model using a pre-trained network (e.g., ResNet18).
    """

    def __init__(self, num_classes=None):
        super(Model3, self).__init__()
        if num_classes is None:
            raise ValueError("num_classes must be specified and cannot be None.")
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # Replace the last fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x  # Output shape: [batch_size, num_classes]

class Model4(nn.Module):
    """
    Model 4: Advanced model using EfficientNet-B0.
    """

    def __init__(self, num_classes=None):
        super(Model4, self).__init__()
        if num_classes is None:
            raise ValueError("num_classes must be specified and cannot be None.")
        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # Replace the classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class Model5(nn.Module):
    """
    Model 5: EfficientNet-B0 with an extended classifier.
    """

    def __init__(self, num_classes=None):
        super(Model5, self).__init__()
        if num_classes is None:
            raise ValueError("num_classes must be specified and cannot be None.")

        # Load pre-trained EfficientNet-B0
        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Optionally freeze feature extractor layers
        # for param in self.model.features.parameters():
        #     param.requires_grad = False

        # Replace the classifier with a custom classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Model6(nn.Module):
    """
    Model 6: Advanced model using EfficientNetV2-S.
    """

    def __init__(self, num_classes=None):
        super(Model6, self).__init__()
        if num_classes is None:
            raise ValueError("num_classes must be specified and cannot be None.")

        # Load pre-trained EfficientNetV2-S
        self.model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

        # Replace the classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class Model7(nn.Module):
    """
    Model7: Advanced image classification model using RegNetX-400MF.

    This model leverages a pre-trained RegNetX-400MF backbone and adapts it
    for custom classification tasks by modifying the final fully connected layer.
    """

    def __init__(self, num_classes=None):
        """
        Initializes the Model7 architecture.

        Args:
            num_classes (int): Number of target classes for classification.
                                Must be specified and cannot be None.

        Raises:
            ValueError: If num_classes is not provided.
        """
        super(Model7, self).__init__()

        if num_classes is None:
            raise ValueError("num_classes must be specified and cannot be None.")

        # Load pre-trained RegNetX-400MF with default ImageNet weights
        self.model = models.regnet_x_800mf(weights=RegNet_X_800MF_Weights.DEFAULT)

        # Replace the classifier (fully connected layer) to match the desired number of classes

        in_features = self.model.fc.in_features

        # Replace the original classifier with a new sequential classifier
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.6, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        x = self.model(x)
        return x



def get_model(model_name, num_classes):
    if model_name == 'model1':
        return Model1(num_classes=num_classes)
    elif model_name == 'model2':
        return Model2(num_classes=num_classes)
    elif model_name == 'model3':
        return Model3(num_classes=num_classes)
    elif model_name == 'model4':
        return Model4(num_classes=num_classes)
    elif model_name == 'model5':
        return Model5(num_classes=num_classes)
    elif model_name == 'model6':
        return Model6(num_classes=num_classes)
    elif model_name == 'model7':
        return Model7(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    # Example usage
    model = get_model('model7', num_classes=5)  # Replace 5 with your desired number of classes
    print(model)
