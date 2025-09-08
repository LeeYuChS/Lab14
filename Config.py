import os
import torch
from torch import nn
from datetime import datetime


class Config():
    root_path = os.getcwd()

    training_batch_size = 16
    training_epoch = 45
    training_LR = 0.0001

    @staticmethod
    def get_optimizer(model):
        return torch.optim.Adam(model.parameters(), lr=config.training_LR)
    training_loss = nn.CrossEntropyLoss()

    vit_base_patch16 = "vit_base_patch16_224"    
    vit_base_patch32 = "vit_base_patch32_224"
    vit_large_patch16 = "vit_large_patch16_224"    
    vit_large_patch32 = "vit_large_patch32_224"
    
    resnet50 = "resnet50"
    resnet101 = "resnet101"
    mobilenet_v2= "mobilenet_v2"
    efficientnet = "efficientnet"

    # model_list_cnn = [efficientnet]
    model_list = [vit_large_patch32]
    # model = "vit_large_patch16_224"
    
    # model = "vit_large_patch32_224"
    # model = "vit_huge_patch14_224"
    # continue_weights = r"F:\20250711_backup\Lab_forfun\Lab14-main\checkpoints\2508201040\best_vit_large_patch32_224_model.pth"
    continue_weights = None
    dataset_type = "stroke"
    num_classes = 3
    # stroke_dataset
    image_path = os.path.join(root_path, "stroke_dataset")
    # valid_image_path = 
    image_size = 224
    save_path = os.path.join(os.getcwd(), 'checkpoints', datetime.now().strftime("%y%m%d%H%M"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = Config()
print(config.root_path)