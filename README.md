# Brain Stroke Classification Framework

A comprehensive deep learning framework for classifying brain stroke images into three categories: Bleeding, Ischemia, and Normal. This project provides multiple pre-trained model architectures with configurable training pipeline and visualization tools.

## Dataset Classes

- **Bleeding**: Hemorrhagic stroke with brain bleeding
- **Ischemia**: Ischemic stroke caused by blocked blood flow  
- **Normal**: Healthy brain scans without stroke indicators

## Supported Models

### Vision Transformer (ViT)
- `vit_base_patch16` - Base ViT with 16x16 patches
- `vit_base_patch32` - Base ViT with 32x32 patches
- `vit_large_patch16` - Large ViT with 16x16 patches
- `vit_large_patch32` - Large ViT with 32x32 patches

### Convolutional Neural Networks
- `resnet50` - ResNet-50 architecture
- `resnet101` - ResNet-101 architecture
- `mobilenetv2` - MobileNetV2 for efficient inference
- `efficientnet` - EfficientNet for balanced accuracy/efficiency

## Key Features

- **Stratified Shuffle Split**: Ensures balanced class distribution across train/validation sets
- **Progressive Training**: Different dataset percentages at various training stages for stable convergence
- **Visualization Tools**: Training/validation plots and Grad-CAM model explanations
- **Configurable Pipeline**: Single configuration file controls entire training process

---

## Requirements

### Python Dependencies
```
python=3.8.20
efficientnet-pytorch=0.7.1
torch=2.4.1+cu118
torchaudio=2.4.1+cu118
torchvision=0.19.1+cu118
`pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118`
```

### Hardware Specifications
- CPU: Intel(R) Core(TM) i9-9900X
- GPU: NVIDIA GeForce RTX 2080 (vram 8GB)

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd lab14-main
   ```

2. **Create virtual environment:**
   ```bash
   conda create lab14 
   
   # Windows
   .\venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Project Structure

```
brain-stroke-classification/
├── config.py                    # Main configuration file
├── train.py                     # Model training script
├── plot.py                      # Training results visualization
├── grad_cam.py                  # Model explanation with Grad-CAM
├── models/
│   ├── vit_models.py           # Vision Transformer implementations
│   ├── cnn_models.py           # CNN model implementations
│   └── model_factory.py       # Model selection factory
├── data/
│   ├── train/
│   │   ├── bleeding/
│   │   ├── ischemia/
│   │   └── normal/
│   ├── validation/
│   │   ├── bleeding/
│   │   ├── ischemia/
│   │   └── normal/
│   └── test/
│       ├── bleeding/
│       ├── ischemia/
│       └── normal/
├── utils/
│   ├── data_loader.py          # Data loading and augmentation
│   ├── training_utils.py       # Training utilities
│   └── visualization.py       # Plotting and visualization functions
├── checkpoints/                # Saved model weights
├── logs/                       # Training logs
├── plots/                      # Generated training plots
└── README.md
```

---

## Configuration

### Key Configuration Parameters

Edit `config.py` to customize your training:

```python
# Model selection
MODEL_NAME = 'vit_base_patch16'  # Choose from supported models

# Dataset configuration
DATASET_PATH = './data'
NUM_CLASSES = 3
CLASS_NAMES = ['bleeding', 'ischemia', 'normal']

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# Stratified split configuration
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.1
RANDOM_SEED = 42

# Progressive training stages
PROGRESSIVE_TRAINING = True
STAGE_PERCENTAGES = [0.3, 0.6, 1.0]  # Gradual dataset increase
STAGE_EPOCHS = [20, 40, 40]          # Epochs per stage

# Image preprocessing
IMAGE_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Paths
CHECKPOINT_PATH = './checkpoints'
LOG_PATH = './logs'
PLOT_PATH = './plots'
```

---

## Usage

### 1. Prepare Your Dataset

Organize your brain stroke images in the following structure:
```
data/
├── bleeding/
├── ischemia/
└── normal/
```

### 2. Configure Training

Modify `config.py` according to your requirements:
- Choose model architecture
- Set dataset paths
- Adjust hyperparameters
- Configure progressive training stages

### 3. Train the Model

```bash
python train.py
```

The training script will:
- Apply stratified shuffle split for balanced datasets
- Execute progressive training if enabled
- Save best model checkpoints
- Generate training logs

### 4. Visualize Results

Generate training/validation plots:
```bash
python plot.py
```

This creates:
- Loss curves
- Accuracy curves
- Confusion matrices
- Class distribution plots

### 5. Model Explanation with Grad-CAM

Generate visual explanations:
```bash
python grad_cam.py --model_path ./checkpoints/best_model.pth --image_path ./test_image.jpg
```

Grad-CAM highlights image regions important for classification decisions.

---

## Progressive Training Strategy

The framework implements progressive training to improve model stability:

1. **Stage 1**: Train on 30% of dataset for initial feature learning
2. **Stage 2**: Expand to 60% of dataset for pattern refincement  
3. **Stage 3**: Use full dataset for final optimization

This approach prevents overfitting and ensures robust feature learning across all classes.

---

## Model Performance Tips

### For Vision Transformers:
- Use larger patch sizes (patch32) for smaller datasets
- Increase batch size when possible
- Consider longer training schedules

### For CNNs:
- ResNet models work well with standard augmentations
- MobileNetV2 for deployment on resource-constrained devices
- EfficientNet provides good accuracy/efficiency balance

### Data Augmentation:
```python
# Recommended augmentations in config.py
AUGMENTATIONS = {
    'rotation': 15,
    'horizontal_flip': True,
    'brightness': 0.2,
    'contrast': 0.2,
    'gaussian_blur': 0.1
}
```

---

## Troubleshooting

**GPU Memory Issues:**
- Reduce batch size in `config.py`
- Use gradient accumulation for effective larger batches
- Choose smaller model variants (base instead of large)

**Poor Convergence:**
- Adjust learning rate (try 1e-5 for fine-tuning)
- Increase early stopping patience
- Check class balance in dataset

**Low Accuracy:**
- Verify correct data preprocessing
- Ensure proper train/validation split
- Consider data augmentation strategies
- Check for data leakage between splits

---

## Output Files

After training completion:
- `./checkpoints/best_model.pth` - Best performing model weights
- `./logs/training.log` - Detailed training logs
- `./plots/training_curves.png` - Loss and accuracy plots  
- `./plots/confusion_matrix.png` - Final model performance
- `./plots/grad_cam_examples/` - Visual explanations

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{brain-stroke-classification,
  title={Brain Stroke Classification Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/brain-stroke-classification}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.