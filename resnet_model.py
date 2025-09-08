import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

# Load pretrained ResNet101
def resnet50(num_classes):
    print("\n" + "="*60)
    print("Training ResNet50")
    print("="*60)
    
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# Load pretrained ResNet101
def resnet101(num_classes):
    print("\n" + "="*60)
    print("Training ResNet101")
    print("="*60)
    
    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

# Load pretrained MobileNetV2
def mobilenet_v2(num_classes):
    print("\n" + "="*60)
    print("Training MobileNetV2")
    print("="*60)
    
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model

# Load pretrained EfficientNet-B0
def efficientnet(num_classes):
    print("\n" + "="*60)
    print("Training EfficientNet-B0")
    print("="*60)
    
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, num_classes)

    return model