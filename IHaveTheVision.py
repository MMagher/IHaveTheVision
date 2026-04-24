import cv2
import easyocr
import os
import sys
import time
import numpy as np
import re
from datetime import datetime

testcase = {
    "e1": "Passport", 
    "e2": "git push -u origin main", 
    "e3": "All exams times are scheduled in EST (eastern Standard Time) Semester Course Title CRN Instructor Date Start Duration Room Location Surname Winter CSCI Artificail 72751 Davoudi April 22 3:30 p.m. 2 hours 15 minutes SIR2060 North 2026 4610U Intelligence",
}

def calculateAccuracy(extracted, expected):
    if not extracted or extracted == "No text detected":
        return 0.0
    
    extracted = str(extracted).strip()
    expected = str(expected).strip()
    
    if extracted == expected:
        return 100.0
    
    maxLen = max(len(extracted), len(expected))
    if maxLen == 0:
        return 0.0
    
    matches = 0
    for i in range(min(len(extracted), len(expected))):
        if extracted[i] == expected[i]:
            matches += 1
    
    charAccuracy = (matches / maxLen) * 100
    
    extractedWords = set(re.findall(r'\b\w+\b', extracted.lower()))
    expectedWords = set(re.findall(r'\b\w+\b', expected.lower()))
    
    if expectedWords:
        wordMatches = len(extractedWords & expectedWords)
        wordAccuracy = (wordMatches / len(expectedWords)) * 100
    else:
        wordAccuracy = 0
    
    combinedAccuracy = (charAccuracy * 0.4) + (wordAccuracy * 0.6)
    
    return round(combinedAccuracy, 2)

def logResult(imageName, extractedText, expectedText, accuracy, processingTime):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logEntry = f"""
{'='*60}
Timestamp: {timestamp}
Image: {imageName}
Processing Time: {processingTime:.2f} seconds
Accuracy: {accuracy}%
Expected Text: {expectedText}
Extracted Text: {extractedText}
{'='*60}
"""
    
    with open("latest-log.txt", "a") as logFile:
        logFile.write(logEntry)
    
    with open("latest-log-summary.txt", "w") as summaryFile:
        summaryFile.write(f"Most Recent Test - {timestamp}\n")
        summaryFile.write(f"Accuracy: {accuracy}%\n")
        summaryFile.write(f"Expected: {expectedText}\n")
        summaryFile.write(f"Extracted: {extractedText}\n")

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

def captureFromWebcam():
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

def detectTestCase(imagePath):
    filename = os.path.basename(imagePath).lower()
    
    if 'test1' in filename or 'e1' in filename:
        return 'e1', testcase['e1']
    elif 'test2' in filename or 'e2' in filename:
        return 'e2', testcase['e2']
    elif 'test3' in filename or 'e3' in filename:
        return 'e3', testcase['e3']
    else:
        return None, None

print("=" * 55)
print("IhaveTheVision")
print("=" * 55)

print("\nInput source:")
print("  1. Image file")
print("  2. Webcam (take a picture)")

sourceChoice = input("\nChoose (1 or 2): ").strip()

imageData = None
imageIdentifier = None
testKey = None
expectedText = None

if sourceChoice == "2":
    print("\n[INFO] Starting webcam...")
    imageData = captureFromWebcam()
    if imageData is None:
        print("Cancelled or webcam error.")
        sys.exit(0)
    imageIdentifier = "webcam_capture"
    print("\n[INFO] Image captured from webcam")
else:
    imageName = input("\nImage filename: ").strip().strip('"').strip("'")
    imageData = imageName
    imageIdentifier = imageName
    
    testKey, expectedText = detectTestCase(imageName)
    if testKey:
        print(f"\n[INFO] Auto-detected test case: {testKey}")
        print(f"[INFO] Expected: {expectedText[:100]}...")

print("\nOptional preprocessing (Choose only one, both gives quack results):")
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

if testKey and expectedText:
    accuracy = calculateAccuracy(result, expectedText)
    print(f"\n{'='*55}")
    print(f"TEST RESULT:")
    print(f"  Test Case: {testKey}")
    print(f"  Expected: {expectedText}")
    print(f"  Accuracy: {accuracy}%")
    print(f"{'='*55}")
    
    logResult(imageIdentifier, result, expectedText, accuracy, elapsed)
    print(f"\n[INFO] Results logged to 'latest-log.txt'")
    
    if accuracy >= 90:
        print("[RESULT] EXCELLENT - High accuracy")
    elif accuracy >= 70:
        print("[RESULT] GOOD - Acceptable accuracy")
    elif accuracy >= 50:
        print("[RESULT] FAIR - Needs improvement")
    else:
        print("[RESULT] POOR - Check image quality")
else:
    with open("latest-log.txt", "a") as logFile:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logFile.write(f"\n{'='*60}\nTimestamp: {timestamp}\nImage: {imageIdentifier}\nExtracted: {result}\nTime: {elapsed:.2f}s\n{'='*60}\n")
    if sourceChoice != "2":
        print("\n[INFO] No test case detected for this filename (looking for 'test1', 'test2', or 'test3')")