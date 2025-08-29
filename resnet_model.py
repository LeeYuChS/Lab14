

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
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model










# import torch
# import torch.nn as nn

# class BasicBlock(nn.Module):
#     expansion = 1
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels * self.expansion:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1,
#                           stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * self.expansion)
#             )

#     def forward(self, x):
#         identity = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += self.shortcut(identity)
#         out = self.relu(out)
#         return out


# class Bottleneck(nn.Module):
#     expansion = 4
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
#                                kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels * self.expansion:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * self.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * self.expansion)
#             )

#     def forward(self, x):
#         identity = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)
#         out += self.shortcut(identity)
#         out = self.relu(out)
#         return out


# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
#         super(ResNet, self).__init__()
#         self.in_channels = 64
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#         # Initialize weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#         # Zero-initialize residual branch BatchNorm (optional, improves training stability)
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)

#     def _make_layer(self, block, out_channels, blocks, stride):
#         layers = []
#         layers.append(block(self.in_channels, out_channels, stride))
#         self.in_channels = out_channels * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.in_channels, out_channels, stride=1))
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x


# # Factory functions
# def resnet18(num_classes=1000, zero_init_residual=False, pretrained = True, continue_weights = None):
#     model = ResNet(BasicBlock, [2, 2, 2, 2], 
#                     num_classes=num_classes,
#                     zero_init_residual=zero_init_residual)
#     if pretrained==True and continue_weights==None:
#         url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth"
#         if url:
#             state_dict = load_state_dict_from_url(url)  # 官方推薦新的API
#         else:
#             raise ValueError(f'Pretrained model for vit_base_patch16_224_in21k has not yet been released')
#         # model.load_state_dict(state_dict, strict=False)
#         load_weights_except_head(model, state_dict)
#         print(f"---------------- Loaded pre-trained weights ----------------")
#         return model
#     elif continue_weights!=None:
#         print(f"---------------- Using continue weights ----------------")
#         state_dict = torch.load(continue_weights)
#         model.load_state_dict(state_dict)
#         return model
#     else:
#         print(f"---------------- No pre-trained weights ----------------")
#         return model

# def resnet34(num_classes=1000, zero_init_residual=False, pretrained: bool = True, continue_weights: str = None):
#     return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes,
#                   zero_init_residual=zero_init_residual)

# def resnet50(num_classes=1000, zero_init_residual=False, pretrained: bool = True, continue_weights: str = None):
#     return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes,
#                   zero_init_residual=zero_init_residual)

# def resnet101(num_classes=1000, zero_init_residual=False, pretrained: bool = True, continue_weights: str = None):
#     return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes,
#                   zero_init_residual=zero_init_residual)

# def resnet152(num_classes=1000, zero_init_residual=False, pretrained: bool = True, continue_weights: str = None):
#     return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes,
#                   zero_init_residual=zero_init_residual)





# # # Flexible version
# # def resnet_custom(block=BasicBlock, depth=[2, 2, 2, 2], num_classes=1000, zero_init_residual=False):
# #     return ResNet(block, depth, num_classes=num_classes, zero_init_residual=zero_init_residual)