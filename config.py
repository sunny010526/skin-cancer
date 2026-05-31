import os
import torch
import platform

# Get base directory of the project (two levels up from this file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    # Flask secret key
    SECRET_KEY = 'skin-cancer-detection-secret-key-2024'

    # Maximum upload size
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

    # Dataset path
    DATA_PATH = r"C:\Users\varap\OneDrive\Desktop\skin\dataset\melanoma_cancer_dataset\train"

    # Paths for model saving and uploads
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'app', 'models', 'saved_models')
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'app', 'static', 'uploads')

    # Device configuration (GPU if available)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Batch size adapted to platform
    if platform.system() == 'Windows':
        BATCH_SIZE = 8
    else:
        BATCH_SIZE = 16

    # Image size
    IMAGE_SIZE = (224, 224)

    # Training hyperparameters
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    PATIENCE = 5

    # Dataset info
    NUM_CLASSES = 2
    CLASS_NAMES = ['benign', 'malignant']

    # Data split ratios
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2

    # Vision Transformer model specific
    VIT_MODEL_NAME = "vit_tiny_patch16_224"
    VIT_PRETRAINED = True

    def __init__(self):
        # Auto-create folders if missing
        os.makedirs(self.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(self.MODEL_SAVE_PATH, exist_ok=True)
