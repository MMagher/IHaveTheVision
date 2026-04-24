import os
import sys
import time
import re
from datetime import datetime
import glob

try:
    import cv2
except ImportError:
    print("OpenCV not installed. Run: pip install opencv-python")
    sys.exit(1)

try:
    import easyocr
except ImportError:
    print("easyOCR not installed. Run: pip install easyocr")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("numpy not installed. Run: pip install numpy")
    sys.exit(1)

testcase = {
    "e1": "PASSPORT", 
    "e2": "git push -u origin main", 
    "e3": "All exams times are scheduled in EST (eastern Standard Time) Semester Course Title CRN Instructor Date Start Duration Room Location Surname Winter CSCI Artificail 72751 Davoudi April 22 3:30 p.m. 2 hours 15 minutes SIR2060 North 2026 4610U Intelligence",
    "e4":"7C Hamilton 8C 3C Mostly  Sunny",
    "e5":"SUNO",
    "e6":"BIKE LANE"
}

if not os.path.exists("logs"):
    os.makedirs("logs")

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

def logToFile(entry):
    with open("logs/latest-log.txt", "a") as logFile:
        logFile.write(entry)

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
    logToFile(logEntry)

def logBatchResult(batchResults):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logEntry = f"""
{'#'*60}
BATCH RUN - {timestamp}
{'#'*60}
"""
    
    totalAccuracy = 0
    for result in batchResults:
        logEntry += f"""
Image: {result['image']}
  Expected: {result['expected']}
  Extracted: {result['extracted']}
  Accuracy: {result['accuracy']}%
  Time: {result['time']:.2f}s
"""
        totalAccuracy += result['accuracy']
    
    avgAccuracy = totalAccuracy / len(batchResults) if batchResults else 0
    logEntry += f"""
{'='*60}
AVERAGE ACCURACY: {avgAccuracy:.2f}%
TOTAL IMAGES: {len(batchResults)}
{'='*60}
"""
    
    logToFile(logEntry)

def logWebcamResult(result, processingTime):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logEntry = f"""
{'='*60}
Timestamp: {timestamp}
Image: webcam_capture
Processing Time: {processingTime:.2f} seconds
Extracted Text: {result}
{'='*60}
"""
    logToFile(logEntry)

def logGeneral(imageName, result, processingTime):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logEntry = f"""
{'='*60}
Timestamp: {timestamp}
Image: {imageName}
Processing Time: {processingTime:.2f} seconds
Extracted Text: {result}
{'='*60}
"""
    logToFile(logEntry)

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
    elif 'test4' in filename or 'e4' in filename:
        return 'e4', testcase['e4']
    elif 'test5' in filename or 'e5' in filename:
        return 'e5', testcase['e5']
    elif 'test6' in filename or 'e6' in filename:
        return 'e6', testcase['e6']
    else:
        return None, None

def findEasyImages():
    images = []
    
    if os.path.exists("easy images"):
        for file in os.listdir("easy images"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                images.append(os.path.join("easy images", file))
    
    images.sort()
    return images

print("=" * 55)
print("IhaveTheVision")
print("=" * 55)

print("\nSelect mode:")
print("  1. Single image")
print("  2. Batch process easy images (test1, test2, test3)")
print("  3. Webcam (take a picture)")

modeChoice = input("\nChoose (1, 2, or 3): ").strip()

if modeChoice == "2":
    easyImages = findEasyImages()
    
    print(f"[INFO] Found {len(easyImages)} images:")
    for img in easyImages:
        print(f"  - {img}")
    
    print("\nOptional preprocessing? (Choose only one, both gives quack results):")
    denoise = input("Apply denoising? (y/n): ").strip().lower() in ['y', 'yes']
    sharpen = input("Apply sharpening? (y/n): ").strip().lower() in ['y', 'yes']
    
    batchResults = []
    totalStart = time.time()
    
    for i, imagePath in enumerate(easyImages, 1):
        print(f"processing: {imagePath}")
        
        testKey, expectedText = detectTestCase(imagePath)
        
        if testKey is None or expectedText is None:
            print(f"  [SKIP] No test case mapping for {imagePath}")
            continue
        
        startTime = time.time()
        result = extractTextFromImage(imagePath, denoise, sharpen)
        elapsed = time.time() - startTime
        
        accuracy = calculateAccuracy(result, expectedText)
        
        batchResults.append({
            'image': imagePath,
            'expected': expectedText,
            'extracted': result,
            'accuracy': accuracy,
            'time': elapsed
        })
        
        print(f"  Expected: {expectedText[:50]}...")
        print(f"  Extracted: {result[:50]}...")
        print(f"  Accuracy: {accuracy}%")
        print(f"  Time: {elapsed:.2f}s")
    
    if batchResults:
        totalElapsed = time.time() - totalStart
        avgAccuracy = sum(r['accuracy'] for r in batchResults) / len(batchResults)
        
        print("\n" + "=" * 55)
        print("BATCH COMPLETE")
        print("=" * 55)
        print(f"Total Images: {len(batchResults)}")
        print(f"Average Accuracy: {avgAccuracy:.2f}%")
        print(f"Total Time: {totalElapsed:.2f} seconds")
        print("=" * 55)
        
        logBatchResult(batchResults)
        print("\n[INFO] Results logged to 'logs/latest-log.txt'")
    else:
        print("\n[ERROR] No valid test images found")

elif modeChoice == "3":
    print("\n[INFO] Starting webcam...")
    imageData = captureFromWebcam()
    if imageData is None:
        print("Cancelled or webcam error.")
        sys.exit(0)
    imageIdentifier = "webcam_capture"
    print("\n[INFO] Image captured from webcam")
    
    print("\nOptional preprocessing (Choose only one, both gives quack results):")
    denoise = input("Apply denoising? (y/n): ").strip().lower() in ['y', 'yes']
    sharpen = input("Apply sharpening? (y/n): ").strip().lower() in ['y', 'yes']
    
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
    
    logWebcamResult(result, elapsed)
    print("\n[INFO] Results logged to 'logs/latest-log.txt'")

else:
    imageName = input("\nImage filename: ").strip().strip('"').strip("'")
    imageData = imageName
    imageIdentifier = imageName
    
    testKey, expectedText = detectTestCase(imageName)
    if testKey:
        print(f"\n[INFO] Auto-detected test case: {testKey}")
        print(f"[INFO] Expected: {expectedText[:100]}...")
    
    print("\nOptional preprocessing? (Choose only one, both gives quack results):")
    denoise = input("Apply denoising? (y/n): ").strip().lower() in ['y', 'yes']
    sharpen = input("Apply sharpening? (y/n): ").strip().lower() in ['y', 'yes']
    
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
        print(f"\n[INFO] Results logged to 'logs/latest-log.txt'")
    else:
        logGeneral(imageIdentifier, result, elapsed)
        print("\n[INFO] Results logged to 'logs/latest-log.txt'")