import os
import shutil
from config import Config

def organize_dataset():
    """Organize dataset into proper class directories"""
    data_path = Config.DATA_PATH
    
    if not os.path.exists(data_path):
        print(f"Dataset path does not exist: {data_path}")
        return
    
    items = os.listdir(data_path)
    
    # Check current structure
    print("Current dataset structure:")
    for item in items:
        item_path = os.path.join(data_path, item)
        if os.path.isdir(item_path):
            images = [f for f in os.listdir(item_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"  📁 {item}: {len(images)} images")
        else:
            if item.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"  📄 {item} (image in root)")
    
    # Ask user how to organize
    print("\nHow would you like to organize the dataset?")
    print("1. Auto-detect from existing folders")
    print("2. Manually specify class folders")
    print("3. Keep current structure")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        auto_organize(data_path, items)
    elif choice == "2":
        manual_organize(data_path, items)
    else:
        print("Keeping current structure.")

def auto_organize(data_path, items):
    """Auto-organize based on existing folders"""
    class_folders = [item for item in items if os.path.isdir(os.path.join(data_path, item))]
    
    if not class_folders:
        print("No class folders found. Please use manual organization.")
        return
    
    print(f"Found class folders: {class_folders}")
    print("Dataset is already organized!")

def manual_organize(data_path, items):
    """Manually organize dataset"""
    # Find all images in root
    root_images = [item for item in items if item.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not root_images:
        print("No images found in root directory.")
        return
    
    print(f"Found {len(root_images)} images in root directory.")
    print("Please create class folders and move images accordingly.")
    print("Suggested structure:")
    print("  dataset/train/benign/")
    print("  dataset/train/melanoma/")
    
    # Create suggested folders
    suggested_folders = ['benign', 'melanoma']
    for folder in suggested_folders:
        folder_path = os.path.join(data_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder_path}")
    
    print(f"\nPlease move your {len(root_images)} images to the appropriate class folders.")
    print("Then run the application again.")

if __name__ == "__main__":
    organize_dataset()