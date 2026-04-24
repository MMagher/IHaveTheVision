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

def extractTextFromImage(imagePath, denoise=False, sharpen=False):
    """Load image, preprocess, and extract text using EasyOCR"""
    
    img = cv2.imread(imagePath)
    if img is None:
        return f"Error: Could not read image '{imagePath}'."
    
    processed = preprocessImage(img, denoise, sharpen)
    
    reader = easyocr.Reader(['en'], verbose=False, gpu=False)
    
    results = reader.readtext(processed, detail=0, paragraph=False)
    
    extractedText = " ".join(results)
    
    return extractedText if extractedText else "No text detected"

    
imageName = input("\nImage filename: ").strip().strip('"').strip("'")
print("\nOptional preprocessing (improves accuracy but adds time):")
denoise = input("Apply denoising? (y/n): ").strip().lower() in ['y', 'yes']
sharpen = input("Apply sharpening? (y/n): ").strip().lower() in ['y', 'yes']
    
if denoise or sharpen:
    print("\n[INFO] Processing with additional filters")
    
print(f"\nAnalyzing '{imageName}'...\n")
    
startTime = time.time()
result = extractTextFromImage(imageName, denoise, sharpen)
elapsed = time.time() - startTime
    
print("-" * 55)
print("EXTRACTED TEXT:")
print("-" * 55)
print(result)
print("-" * 55)
print(f"\nTime: {elapsed:.2f} seconds")