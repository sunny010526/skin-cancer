import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.training.trainer import Trainer
from app.utils.data_loader import get_data_loaders
from app.models.vit_model import create_vit_model
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config

def train_standalone():
    """Standalone training script"""
    print("Starting standalone training...")
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders()
    
    # Create model
    model = create_vit_model()
    model = model.to(Config.DEVICE)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # Create trainer and train
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler)
    best_acc = trainer.train(Config.NUM_EPOCHS)
    
    print(f"Training completed! Best accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train_standalone()