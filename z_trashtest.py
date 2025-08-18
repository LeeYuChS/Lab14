


from torch.hub import load_state_dict_from_url
import torch
from transformers import ViTConfig, ViTModel

url = "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16.pth"
if url:
    state_dict = load_state_dict_from_url(url)  # 官方推薦新的API
else:
    raise ValueError(f'Pretrained model for vit_base_patch16_224_in21k has not yet been released')

configuration = ViTConfig()
model = ViTModel(configuration)
# model.load_state_dict(state_dict)
model = torch.load(state_dict)
# print(state_dict)


# from transformers import ViTConfig, ViTModel

# # Initializing a ViT vit-base-patch16-224 style configuration
# configuration = ViTConfig()

# # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
# model = ViTModel(configuration)

# # Accessing the model configuration
# configuration = model.config