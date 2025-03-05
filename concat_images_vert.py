import os
import sys
from PIL import Image
import re

def extract_integer(filename):
    match = re.search(r'(\d+)\.jpg$', filename)
    return int(match.group(1)) if match else -1

def concat_images(folder_path):
    # Get all jpg files in the folder
    images = [f for f in os.listdir(folder_path) if f.endswith('.jpg') and extract_integer(f) != -1]
    
    print(images)
    
    # Sort images by the integer value in their file names
    images.sort(key=extract_integer)
    
    # Open images and store them in a list
    image_list = [Image.open(os.path.join(folder_path, img)) for img in images]
    
    # Get the maximum width and the total height of the concatenated image
    max_width = max(img.width for img in image_list)
    total_height = sum(img.height for img in image_list)
    
    # Create a new blank image with the calculated dimensions
    concat_image = Image.new('RGB', (max_width, total_height))
    
    # Paste each image into the concatenated image
    y_offset = 0
    for img in image_list:
        concat_image.paste(img, (0, y_offset))
        y_offset += img.height
    
    # Save the concatenated image
    concat_image.save(os.path.join(folder_path, 'concat_vert.jpg'))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python concat_images.py <folder_path>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    concat_images(folder_path)