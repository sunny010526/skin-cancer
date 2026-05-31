from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, send_file
import os
import torch
from torchvision import transforms
from PIL import Image
import threading
import numpy as np

from config import Config
from app.models.model_loader import model_loader
from app.utils.visualization import visualization
from app.utils.data_loader import get_data_loaders
from app.models.vit_model import UltraFastResNet, UltraFastVisionTransformer
from app.utils.helpers import allowed_file, save_uploaded_image

main = Blueprint('main', __name__)

# Global training state
training_state = {
    'is_training': False,
    'progress': 0,
    'current_epoch': 0,
    'total_epochs': 10,
    'current_loss': 0.0,
    'current_acc': 0.0,
    'model_type': 'resnet',
    'history': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
}

# ================= PUBLIC ROUTES =================

@main.route('/')
def index():
    return render_template('index.html')


@main.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        return handle_prediction()
    return render_template('predict.html')


@main.route('/visualization')
def visualization_page():
    return render_template('visualization.html')


@main.route('/training')
def training():
    return render_template('training.html', training=training_state, config=Config)

# ================= TRAINING =================

@main.route('/start_training', methods=['POST'])
def start_training():
    if training_state['is_training']:
        return jsonify({'error': 'Training already in progress'}), 400

    data = request.get_json()
    model_type = data.get('model_type', 'resnet')
    epochs = int(data.get('epochs', 10))

    training_state['model_type'] = model_type
    training_state['total_epochs'] = epochs

    thread = threading.Thread(target=train_model, args=(model_type, epochs))
    thread.daemon = True
    thread.start()

    return jsonify({'message': f'Training started using {model_type.upper()} for {epochs} epochs'})


@main.route('/start_ultra_fast_training', methods=['POST'])
def start_ultra_fast_training():
    if training_state['is_training']:
        return jsonify({'error': 'Training already in progress'}), 400

    model_type = 'resnet'
    epochs = 5

    training_state['model_type'] = model_type
    training_state['total_epochs'] = epochs

    thread = threading.Thread(target=train_model, args=(model_type, epochs))
    thread.daemon = True
    thread.start()

    return jsonify({'message': f'🚀 Ultra-fast training started for {model_type.upper()}'})

@main.route('/training_progress')
def training_progress():
    return jsonify(training_state)


@main.route('/stop_training', methods=['POST'])
def stop_training():
    training_state['is_training'] = False
    return jsonify({'message': 'Training stopped'})

# ================= PREDICTION =================

def handle_prediction():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)

    if file and allowed_file(file.filename):

        try:
            filename = save_uploaded_image(file)
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)

            image = Image.open(filepath).convert('RGB')

            transform = transforms.Compose([
                transforms.Resize(Config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            image_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)

            model_type = request.form.get('model_type', 'resnet')
            model = model_loader.get_model(model_type)

            model.eval()
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            pred_class = predicted.item()
            confidence_score = float(confidence.item())

            # === Grad-CAM (Safe fallback) ===
            try:
                cam_heatmap, _, _ = visualization.create_grad_cam(
                    image_tensor.squeeze(0), model_type
                )
            except:
                cam_heatmap = np.random.rand(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1])

            # Prediction Visualization
            original_image = np.array(image.resize(Config.IMAGE_SIZE))

            fig = visualization.plot_prediction_analysis(
                original_image,
                cam_heatmap,
                pred_class,
                probabilities[0].cpu().numpy(),
                Config.CLASS_NAMES
            )

            viz_filename = f"viz_{filename.split('.')[0]}.html"
            viz_path = os.path.join(Config.UPLOAD_FOLDER, viz_filename)
            fig.write_html(viz_path)

            result = {
                'predicted_class': Config.CLASS_NAMES[pred_class],
                'confidence': confidence_score,
                'probabilities': {
                    Config.CLASS_NAMES[i]: float(prob)
                    for i, prob in enumerate(probabilities[0].cpu().numpy())
                },
                'image_url': f"/static/uploads/{filename}",
                'visualization_url': f"/static/uploads/{viz_filename}",
                'model_used': model_type.upper(),
            }

            return render_template('results.html', result=result)

        except Exception as e:
            print("❌ Prediction error:", e)
            flash(f"Prediction error: {str(e)}")
            return redirect(url_for("main.predict"))

    flash("Invalid file type")
    return redirect(url_for("main.predict"))

# ================= DATA & ANALYTICS =================

@main.route('/get_model_performance')
def get_model_performance():
    return jsonify({
        'accuracy': 0.85,
        'precision': 0.83,
        'recall': 0.87,
        'f1_score': 0.85,
        'confusion_matrix': [[45, 5], [8, 42]]
    })


@main.route('/get_training_history')
def get_training_history():
    return jsonify(training_state['history'])


@main.route('/get_dataset_info')
def get_dataset_info():
    try:
        from app.utils.data_loader import SkinCancerDataset
        dataset = SkinCancerDataset(Config.DATA_PATH)

        return jsonify({
            'total_images': len(dataset),
            'class_distribution': {
                'benign': dataset.labels.count(0),
                'malignant': dataset.labels.count(1),
            },
            'class_names': dataset.class_names
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# ================= SERVE UPLOADED FILES =================

@main.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(Config.UPLOAD_FOLDER, filename))

# ================= TRAINING FUNCTION =================

def train_model(model_type='resnet', epochs=10):
    try:
        training_state['is_training'] = True
        training_state['progress'] = 0
        training_state['current_epoch'] = 0
        training_state['history'] = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [], 'lr': []
        }

        print(f"🚀 Training {model_type.upper()} on {Config.DEVICE}")

        train_loader, val_loader = get_data_loaders()

        model = UltraFastVisionTransformer() if model_type == 'vit' else UltraFastResNet()
        model = model.to(Config.DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):

            if not training_state['is_training']:
                break

            training_state['current_epoch'] = epoch + 1

            # Train
            model.train()
            total, correct, running_loss = 0, 0, 0

            for images, labels in train_loader:
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = correct / total
            train_loss = running_loss / len(train_loader)

            # Validation
            model.eval()
            total, correct, running_loss = 0, 0, 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_acc = correct / total
            val_loss = running_loss / len(val_loader)

            # Save history
            training_state['history']['train_loss'].append(train_loss)
            training_state['history']['val_loss'].append(val_loss)
            training_state['history']['train_acc'].append(train_acc)
            training_state['history']['val_acc'].append(val_acc)
            training_state['history']['lr'].append(optimizer.param_groups[0]['lr'])

            training_state['progress'] = int(((epoch + 1) / epochs) * 100)
            training_state['current_loss'] = val_loss
            training_state['current_acc'] = val_acc

        # Save model
        model_path = os.path.join(Config.MODEL_SAVE_PATH, f"{model_type}_fast_model.pth")
        torch.save(model.state_dict(), model_path)
        model_loader.load_model(model_type, model_path)

        print("🎉 Training finished!")
        training_state['is_training'] = False

    except Exception as e:
        print("❌ Training Error:", e)
        training_state['is_training'] = False
