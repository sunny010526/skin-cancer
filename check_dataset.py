import os
from config import Config

def check_dataset_structure():
    data_path = Config.DATA_PATH
    print(f"Checking dataset structure at: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset path does not exist: {data_path}")
        return False
    
    print(f"✓ Dataset path exists: {data_path}")
    
    # List all items in the dataset directory
    items = os.listdir(data_path)
    print(f"Items in dataset directory: {items}")
    
    # Check for class subdirectories
    class_dirs = [d for d in items if os.path.isdir(os.path.join(data_path, d))]
    print(f"Found subdirectories (potential classes): {class_dirs}")
    
    # Check for image files directly in the root
    image_files = [f for f in items if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(f"Found image files in root: {len(image_files)}")
    
    # Check each class directory
    for class_name in class_dirs:
        class_path = os.path.join(data_path, class_name)
        class_images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        print(f"Class '{class_name}': {len(class_images)} images")
    
    return True

if __name__ == "__main__":
    check_dataset_structure()