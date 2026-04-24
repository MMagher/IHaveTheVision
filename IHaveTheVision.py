import cv2
import easyocr
import os
import sys
import time
import numpy as np

def preprocessImage(image, denoise=False, sharpen=False):
    """Apply preprocessing with optional denoise and sharpen"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    if h < 300:
        scale = 300 / h
        newW = int(w * scale)
        gray = cv2.resize(gray, (newW, 300), interpolation=cv2.INTER_CUBIC)
    
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    result = binary
    
    if denoise:
        result = cv2.medianBlur(result, 3)
    
    if sharpen:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        result = cv2.filter2D(result, -1, kernel)
    
    return result

def captureFromWebcam():
    """Capture a single image from webcam"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return None
    
    print("\n[INFO] Webcam opened. Press SPACE to capture, ESC to cancel.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        cv2.imshow("Webcam - Press SPACE to capture", frame)
        
        key = cv2.waitKey(1)
        if key == 32:
            capturedImage = frame
            break
        elif key == 27:
            capturedImage = None
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return capturedImage

def extractTextFromImage(image, denoise=False, sharpen=False):
    """Extract text from image (can be file path or numpy array)"""
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            return f"Error: Could not read image '{image}'."
    else:
        img = image
    
    processed = preprocessImage(img, denoise, sharpen)
    
    reader = easyocr.Reader(['en'], verbose=False, gpu=False)
    
    results = reader.readtext(processed, detail=0, paragraph=False)
    
    extractedText = " ".join(results)
    
    return extractedText if extractedText else "No text detected"

print("\nInput source:")
print("  1. Image file")
print("  2. Webcam (take a picture)")
    
sourceChoice = input("\nChoose (1 or 2): ").strip()
    
imageData = None
    
if sourceChoice == "2":
    print("\n[INFO] Starting webcam...")
    imageData = captureFromWebcam()
    if imageData is None:
        print("Cancelled or webcam error.")
        sys.exit(0)
    print("\n[INFO] Image captured from webcam")
else:
    imageName = input("\nImage filename: ").strip().strip('"').strip("'")
    imageData = imageName
    
print("\nOptional preprocessing Choose only one, both gives quack results:")
denoise = input("Apply denoising? (y/n): ").strip().lower() in ['y', 'yes']
sharpen = input("Apply sharpening? (y/n): ").strip().lower() in ['y', 'yes']
    
if denoise or sharpen:
    print("\n[INFO] Processing with additional filters")

print(f"\nAnalyzing...\n")
    
startTime = time.time()
result = extractTextFromImage(imageData, denoise, sharpen)
elapsed = time.time() - startTime
    
print("-" * 55)
print("EXTRACTED TEXT:")
print("-" * 55)
print(result)
print("-" * 55)
print(f"\nTime: {elapsed:.2f} seconds")