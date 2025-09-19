import numpy as np
from PIL import Image
import time
from tqdm import tqdm

from image_python import python_grayscale, gil_grayscale
from image_cython import grayscale_cython

# Simple color constants to avoid large color blocks
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header():
    """Print a beautified header for the benchmark."""
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}    🚀 IMAGE PROCESSING BENCHMARK 🚀{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.END}")
    print()

def print_section(section_num, title, color=Colors.YELLOW):
    """Print a beautified section header."""
    print(f"\n{Colors.BOLD}{color}[{section_num}] {title}{Colors.END}")
    print(f"{color}{'-' * (len(title) + 6)}{Colors.END}")

def print_result(method, duration, color=Colors.GREEN):
    """Print a beautified result."""
    print(f"{color}    ✓ {method} completed in {Colors.BOLD}{duration:.4f} seconds{Colors.END}")

def print_comparison_table(results):
    """Print a beautified comparison table."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}📊 PERFORMANCE COMPARISON{Colors.END}")
    print(f"{Colors.HEADER}{'=' * 50}{Colors.END}")
    print(f"{Colors.CYAN}Time bars show relative duration (longer = slower){Colors.END}")
    print()
    
    # Find the maximum duration for proper scaling
    max_duration = max(results.values())
    
    for method, duration in results.items():
        # Scale bars based on the maximum duration (30 chars max)
        bar_length = int((duration / max_duration) * 30)
        bar = '█' * bar_length
        print(f"{method:20} {Colors.BOLD}{duration:8.4f}s{Colors.END} {Colors.BLUE}{bar}{Colors.END}")
    
    # Calculate and display speedups
    if len(results) > 1:
        baseline = list(results.values())[0]
        print(f"\n{Colors.BOLD}{Colors.CYAN}🏃 SPEEDUP ANALYSIS{Colors.END}")
        print(f"{Colors.CYAN}{'-' * 30}{Colors.END}")
        
        for i, (method, duration) in enumerate(results.items()):
            if i == 0:
                print(f"{method:20} {Colors.BOLD}baseline{Colors.END}")
            else:
                speedup = baseline / duration
                speedup_color = Colors.GREEN if speedup > 1 else Colors.RED
                improvement_text = "faster" if speedup > 1 else "slower"
                print(f"{method:20} {speedup_color}{Colors.BOLD}{speedup:6.2f}x {improvement_text}{Colors.END}")

def run_with_progress(func, description, *args, **kwargs):
    """Run a function with a progress bar."""
    with tqdm(total=100, desc=description, bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}') as pbar:
        result = func(*args, **kwargs)
        pbar.update(100)
    return result

# --- Main script ---
if __name__ == "__main__":
    # Using a different image name from your traceback
    IMAGE_PATH = "Curiosity_Self-Portrait_at_'Big_Sky'_Drilling_Site.jpg" 
    
    print_header()
    
    # Store results for comparison
    results = {}
    
    # --- 1. Pure Python Version ---
    print_section(1, f"Processing '{IMAGE_PATH}' with Pure Python", Colors.YELLOW)
    gray_img_py, duration_py = run_with_progress(
        python_grayscale, 
        "Pure Python Processing", 
        IMAGE_PATH
    )
    if gray_img_py:
        print_result("Pure Python", duration_py, Colors.GREEN)
        gray_img_py.save("result_python.jpg")
        print(f"    💾 Result saved to 'result_python.jpg'\n")
        results["Pure Python"] = duration_py
    
    # --- 2. Threaded Python Version ---
    print_section(2, f"Processing '{IMAGE_PATH}' with Threaded Python (4 threads)", Colors.YELLOW)
    gray_img_threaded, duration_threaded = run_with_progress(
        gil_grayscale,
        "Threaded Python Processing",
        IMAGE_PATH
    )
    if gray_img_threaded:
        print_result("Threaded Python", duration_threaded, Colors.GREEN)
        gray_img_threaded.save("result_python_threaded.jpg")
        print(f"    💾 Result saved to 'result_python_threaded.jpg'\n")
        results["Threaded Python"] = duration_threaded
    
    # --- 3. Cython Version ---
    print_section(3, f"Processing '{IMAGE_PATH}' with Cython + OpenMP", Colors.YELLOW)
    try:
        img = Image.open(IMAGE_PATH)
        
        img_data = np.array(img, dtype=np.uint8) 
        # ------------------------------------

        height, width, _ = img_data.shape
        
        gray_data_cy = np.zeros((height, width), dtype=np.uint8)
        
        # Run Cython processing with progress bar
        with tqdm(total=100, desc="Cython Processing", bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}') as pbar:
            start_time = time.time()
            grayscale_cython(img_data, gray_data_cy) 
            end_time = time.time()
            pbar.update(100)
        
        duration_cy = end_time - start_time
        
        Image.fromarray(gray_data_cy).save("result_cython.jpg")
        print_result("Cython + OpenMP", duration_cy, Colors.GREEN)
        print(f"    💾 Result saved to 'result_cython.jpg'\n")
        results["Cython + OpenMP"] = duration_cy
        
        # --- 4. Comparison ---
        if results:
            print_comparison_table(results)
            
            print(f"\n{Colors.BOLD}{Colors.GREEN}🎉 Benchmark Complete!{Colors.END}")
            print(f"{Colors.GREEN}All processed images have been saved to the current directory.{Colors.END}")

    except FileNotFoundError:
        print(f"{Colors.RED}❌ Error: Could not find '{IMAGE_PATH}' for the benchmark.{Colors.END}")
        print(f"{Colors.YELLOW}Please ensure the image file is in the same directory.{Colors.END}")
    except ImportError:
        print(f"{Colors.RED}❌ Error: Cython module not found. Did you run the build command?{Colors.END}")
        print(f"{Colors.YELLOW}--> python setup_image.py build_ext --inplace{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}❌ Unexpected error: {e}{Colors.END}")
