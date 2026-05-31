from app import create_app

# Create and initialize the Flask application
app = create_app()

# Print configuration info
print("🚀 Skin Cancer Detection Application - ULTRA-FAST VERSION")
print(f"📁 Upload folder: {app.config.get('UPLOAD_FOLDER')}")
print(f"💾 Model save path: {app.config.get('MODEL_SAVE_PATH')}")
print(f"⚡ Device: {app.config.get('DEVICE')}")
print(f"🖼️  Image size: {app.config.get('IMAGE_SIZE')}")
print(f"📦 Batch size: {app.config.get('BATCH_SIZE')}")

if __name__ == '__main__':
    print("🎯 Starting Skin Cancer Detection Flask Application...")
    print("🌐 Visit http://localhost:5000 to access the application")
    print("💡 Use 'ULTRA-FAST' training for quick results!")
    
    # Run Flask app publicly on all interfaces
    app.run(debug=True, host='0.0.0.0', port=5000)
