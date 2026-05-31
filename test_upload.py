import os
import requests

def test_upload():
    # Test the upload endpoint
    url = 'http://localhost:5000/predict'
    
    # Create a test image (you can use any image file)
    test_image_path = 'test_image.jpg'
    
    if not os.path.exists(test_image_path):
        print("Please create a test image named 'test_image.jpg' in the current directory")
        return
    
    files = {'file': open(test_image_path, 'rb')}
    
    try:
        response = requests.post(url, files=files)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:500]}...")  # First 500 chars
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_upload()