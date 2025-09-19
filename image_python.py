import numpy as np
from PIL import Image
import time
import threading
from tqdm import tqdm

def python_grayscale(image_path):
    """Loads an image and converts it to grayscale using a pure Python loop."""
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
        print("Please download a large JPG image and save it as 'sample_image.jpg'")
        return None, 0

    img_data = np.asarray(img, dtype=np.uint8)
    height, width, _ = img_data.shape
    gray_data = np.zeros((height, width), dtype=np.uint8)
    
    start_time = time.time()

    with tqdm(total=height, desc="Processing rows", leave=False) as pbar:
        for y in range(height):
            for x in range(width):
                r, g, b = img_data[y, x]
                gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
                gray_data[y, x] = gray_value
            
            if y % 10 == 0 or y == height - 1:
                pbar.update(min(10, height - pbar.n))
            
    end_time = time.time()
    
    gray_image = Image.fromarray(gray_data)
    return gray_image, (end_time - start_time)

def gil_grayscale(image_path, num_threads=4):
    """Loads an image and converts it to grayscale using multiple threads."""
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
        print("Please download a large JPG image and save it as 'sample_image.jpg'")
        return None, 0

    img_data = np.asarray(img, dtype=np.uint8)
    height, width, _ = img_data.shape
    gray_data = np.zeros((height, width), dtype=np.uint8)
    
    start_time = time.time()

    completed_rows = [0]
    total_rows = height
    
    pbar = tqdm(total=total_rows, desc="Processing with threads", leave=False)

    def process_rows(start_row, end_row):
        """Process a range of rows for grayscale conversion."""
        for y in range(start_row, end_row):
            for x in range(width):
                r, g, b = img_data[y, x]
                gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
                gray_data[y, x] = gray_value
            
            completed_rows[0] += 1
            if completed_rows[0] % 10 == 0:
                pbar.update(10)

    rows_per_thread = height // num_threads
    threads = []
    
    for i in range(num_threads):
        start_row = i * rows_per_thread
        end_row = (i + 1) * rows_per_thread if i < num_threads - 1 else height
        
        thread = threading.Thread(target=process_rows, args=(start_row, end_row))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    pbar.update(total_rows - pbar.n)
    pbar.close()
            
    end_time = time.time()
    
    gray_image = Image.fromarray(gray_data)
    return gray_image, (end_time - start_time)

if __name__ == "__main__":
    IMAGE_PATH = "Curiosity_Self-Portrait_at_'Big_Sky'_Drilling_Site.jpg"
    
    print(f"Processing '{IMAGE_PATH}' with pure Python...")
    gray_img, duration = python_grayscale(IMAGE_PATH)
    
    if gray_img:
        gray_img.save("result_python.jpg")
        print(f"Python version took: {duration:.4f} seconds.")
        print("Result saved to 'result_python.jpg'")
    
    print(f"\nProcessing '{IMAGE_PATH}' with threaded Python...")
    gray_img_threaded, duration_threaded = gil_grayscale(IMAGE_PATH)
    
    if gray_img_threaded:
        gray_img_threaded.save("result_python_threaded.jpg")
        print(f"Threaded Python version took: {duration_threaded:.4f} seconds.")
        print("Result saved to 'result_python_threaded.jpg'")
        
        if duration > 0:
            speedup = duration / duration_threaded
            print(f"Speedup: {speedup:.2f}x")
