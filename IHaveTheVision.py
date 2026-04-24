import cv2
import easyocr
import os
import sys
import time
import numpy as np
from paddleocr import PaddleOCR

def preprocessImage(image, denoise=False, sharpen=False):
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

def extractTextEasyOcr(imagePath, denoise=False, sharpen=False):
    img = cv2.imread(imagePath)
    processed = preprocessImage(img, denoise, sharpen)
    reader = easyocr.Reader(['en'], verbose=False, gpu=False)
    results = reader.readtext(processed, detail=0, paragraph=False)
    return " ".join(results) if results else "No text detected"

def extractTextPaddleOcr(imagePath, denoise=False, sharpen=False):
    img = cv2.imread(imagePath)
    processed = preprocessImage(img, denoise, sharpen)
    
    tempPath = "temp_processed.png"
    cv2.imwrite(tempPath, processed)
    
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    result = ocr.ocr(tempPath, cls=True)
    
    os.remove(tempPath)
    
    if not result or not result[0]:
        return "No text detected"
    
    texts = [line[1][0] for line in result[0]]
    return " ".join(texts)

def extractTextFromImage(imagePath, engine="easyocr", denoise=False, sharpen=False):
    if not os.path.exists(imagePath):
        return f"Error: File '{imagePath}' not found."
    
    if engine == "easyocr":
        return extractTextEasyOcr(imagePath, denoise, sharpen)
    elif engine == "paddleocr":
        return extractTextPaddleOcr(imagePath, denoise, sharpen)
    else:
        return f"Error: Unknown engine '{engine}'"

def main():
    print("=" * 55)
    print("IhaveTheVision")
    print("=" * 55)
    
    imageName = input("\nImage filename: ").strip().strip('"').strip("'")
    
    print("\nOCR Engine:")
    print("  1. EasyOCR (faster)")
    print("  2. PaddleOCR (slower but more accurate)")
    
    engineChoice = input("\nUse PaddleOCR? (y/n): ").strip().lower()
    
    if engineChoice in ['y', 'yes']:
        engine = "paddleocr"
        print("\n[INFO] Using PaddleOCR")
    else:
        engine = "easyocr"
        print("\n[INFO] Using EasyOCR")
    
    print("\nOptional preprocessing (can improve quality but adds time):")
    denoise = input("Apply denoising? (y/n): ").strip().lower() in ['y', 'yes']
    sharpen = input("Apply sharpening? (y/n): ").strip().lower() in ['y', 'yes']
    
    if denoise or sharpen:
        print("\n[INFO] Processing with additional filters...")
    
    print(f"\nAnalyzing '{imageName}'...\n")
    
    startTime = time.time()
    result = extractTextFromImage(imageName, engine, denoise, sharpen)
    elapsed = time.time() - startTime
    
    print("-" * 55)
    print("EXTRACTED TEXT:")
    print("-" * 55)
    print(result)
    print("-" * 55)
    print(f"\nTime: {elapsed:.2f} seconds")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)