from flask import Flask
import os
from config import Config

def create_app():
    app = Flask(__name__)

    # Load config from app/config.py
    app.config.from_object(Config)

    # Create required directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_SAVE_PATH'], exist_ok=True)

    # Initialize model loader + visualization
    from app.models.model_loader import model_loader
    from app.utils.visualization import visualization
    visualization.model_loader = model_loader

    # Load pre-trained model if available
    try:
        model_path = os.path.join(app.config['MODEL_SAVE_PATH'], 'resnet_fast_model.pth')

        if os.path.exists(model_path):
            model_loader.load_model('resnet', model_path)
            print("✓ Pre-trained model loaded successfully")
        else:
            print("ℹ No pre-trained model found. Train a model first.")
    except Exception as e:
        print(f"✗ Error loading model: {e}")

    # Register routes
    from app.routes import main
    app.register_blueprint(main)

    return app
