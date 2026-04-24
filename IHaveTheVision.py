import cv2
import easyocr
import os
import sys

def preprocess_image(image):
    """Apply preprocessing to improve OCR accuracy"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize if too small (minimum height 300px)
    h, w = gray.shape
    if h < 300:
        scale = 300 / h
        new_w = int(w * scale)
        gray = cv2.resize(gray, (new_w, 300), interpolation=cv2.INTER_CUBIC)
    
    # Apply adaptive thresholding (works well for varying lighting)
    binary = cv2.adaptiveThreshold(gray, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Remove small noise
    denoised = cv2.medianBlur(binary, 3)
    
    return denoised

def extract_text_from_image(image_path):
    """Load image, process it, and extract text"""
    # Check if file exists
    if not os.path.exists(image_path):
        return f"Error: File '{image_path}' not found."
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return f"Error: Could not read image '{image_path}'. Make sure it's a valid image file."
    
    # Preprocess
    processed = preprocess_image(img)
    
    # Initialize OCR reader (only once, but we'll do it here for simplicity)
    reader = easyocr.Reader(['en'])
    
    # Extract text
    results = reader.readtext(processed, detail=0, paragraph=False)
    
    # Join all detected text
    extracted_text = " ".join(results)
    
    return extracted_text if extracted_text else "No text detected in the image."

def main():
    print("=" * 50)
    print("Text Extractor from Image")
    print("=" * 50)
    
    # Ask for image filename
    image_name = input("\nEnter the image filename (e.g., photo.jpg or C:/folder/image.png): ").strip()
    
    # Remove quotes if user added them
    image_name = image_name.strip('"').strip("'")
    
    print(f"\nAnalyzing '{image_name}'...\n")
    
    # Extract text
    result = extract_text_from_image(image_name)
    
    # Print results
    print("-" * 50)
    print("EXTRACTED TEXT:")
    print("-" * 50)
    print(result)
    print("-" * 50)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("\nMake sure you have installed required packages:")
        print("pip install opencv-python easyocr torch")
        sys.exit(1)