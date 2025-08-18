
import os
import torch
import shutil
import json
import numpy as np
from matplotlib import pyplot as plt
from Config import config
from torch import nn
from Model import (vit_base_patch16_224_in21k,
                   vit_base_patch32_224_in21k,
                   vit_large_patch16_224_in21k,
                   vit_large_patch32_224_in21k,
                   vit_huge_patch14_224_in21k)


def set_seed(seed):
    """
    set random seed
    Args:
        seed
    Returns: None

    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True




def create_model(model, num_classes):
    print("using {} model.".format(model))
    if model == "vit_base_patch16_224":
        model = vit_base_patch16_224_in21k(num_classes, has_logits=False)
    elif model == "vit_base_patch32_224":
        model = vit_base_patch32_224_in21k(num_classes, has_logits=False)
    elif model == "vit_large_patch16_224":
        model = vit_large_patch16_224_in21k(num_classes, has_logits=False)
    elif model == "vit_large_patch32_224":
        model = vit_large_patch32_224_in21k(num_classes, has_logits=False)
    elif model == "vit_huge_patch14_224":
        model = vit_huge_patch14_224_in21k(num_classes, has_logits=False)
    
    # elif model == "vit_base_patch16_224_pretrained":
    #     model == vit_base_patch16_224_in21k(num_classes, has_logits=False)
    # elif model == "vit_base_patch32_224_pretrained":
    #     model = vit_base_patch32_224_in21k(num_classes, has_logits=False)
    # elif model == "vit_large_patch16_224_pretrained":
    #     model = vit_large_patch16_224_in21k(num_classes, has_logits=False)
    # elif model == "vit_large_patch32_224_pretrained":
    #     model = vit_large_patch32_224_in21k(num_classes, has_logits=False)
    # elif model == "vit_huge_patch14_224_pretrained":
    #     model = vit_huge_patch14_224_in21k(num_classes, has_logits=False)

    else:
        raise Exception("Can't find any model name call {}".format(model))

    return model



def model_parallel(args, model):
    device_ids = [i for i in range(len(args.gpu.split(',')))]
    model = nn.DataParallel(model, device_ids=device_ids)
    return model




def save_history_json(history, filepath):
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=4)


def plot_history(history, model_name):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12,5))

    plt.subplot(1,3,1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['valid_loss'], label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
###########################################################################
    plt.subplot(1,3,2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['valid_acc'], label='Valid Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
###########################################################################
    plt.subplot(1,3,3)
    plt.plot(epochs, history['valid_precision'], label='Valid Precision')
    plt.plot(epochs, history['valid_recall'], label='Valid Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Precision or Recall')
    plt.legend()
###########################################################################
    # plt.subplot(1,4,2)
    # plt.plot(epochs, history['train_acc'], label='Train Acc')
    # plt.plot(epochs, history['valid_acc'], label='Valid Acc')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
###########################################################################
    plt.tight_layout()
    plt.savefig(os.path.join(config.save_path, f"{model_name}.jpg"))
    plt.close()
    # plt.show()



def remove_dir_and_create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(dir_name, "Creat OK")
    else:
        shutil.rmtree(dir_name)
        os.makedirs(dir_name)
        print(dir_name, "Remove and Creat OK")