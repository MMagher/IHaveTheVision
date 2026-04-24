
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

# Test cases dictionary for validation
testcase = {
    "e1": "PASSPORT", 
    "e2": "git push -u origin main", 
    "e3": "All exams times are scheduled in EST (eastern Standard Time) Semester Course Title CRN Instructor Date Start Duration Room Location Surname Winter CSCI Artificail 72751 Davoudi April 22 3:30 p.m. 2 hours 15 minutes SIR2060 North 2026 4610U Intelligence",
    "e4":"7C Hamilton 8C 3C Mostly  Sunny",
    "e5":"SUNO",
    "e6":"BIKE LANE"
}

def calculateAccuracy(extracted, expected):
    """Calculate accuracy percentage between extracted and expected text using character and word matching"""
    if not extracted or extracted == "No text detected":
        return 0.0
    
    extracted = str(extracted).strip()
    expected = str(expected).strip()
    
    # Exact match returns 100%
    if extracted == expected:
        return 100.0
    
    maxLen = max(len(extracted), len(expected))
    if maxLen == 0:
        return 0.0
    
    # Character-level accuracy
    matches = 0
    for i in range(min(len(extracted), len(expected))):
        if extracted[i] == expected[i]:
            matches += 1
    
    charAccuracy = (matches / maxLen) * 100
    
    # Word-level accuracy
    extractedWords = set(re.findall(r'\b\w+\b', extracted.lower()))
    expectedWords = set(re.findall(r'\b\w+\b', expected.lower()))
    
    if expectedWords:
        wordMatches = len(extractedWords & expectedWords)
        wordAccuracy = (wordMatches / len(expectedWords)) * 100
    else:
        wordAccuracy = 0
    
    # Combined accuracy (40% character, 60% word)
    combinedAccuracy = (charAccuracy * 0.4) + (wordAccuracy * 0.6)
    
    return round(combinedAccuracy, 2)

def logResult(imageName, extractedText, expectedText, accuracy, processingTime):
    """Log single image test results to file"""
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
    
    # Append to main log
    with open("latest-log.txt", "a") as logFile:
        logFile.write(logEntry)
    
    # Overwrite summary with most recent test
    with open("latest-log-summary.txt", "w") as summaryFile:
        summaryFile.write(f"Most Recent Test - {timestamp}\n")
        summaryFile.write(f"Accuracy: {accuracy}%\n")
        summaryFile.write(f"Expected: {expectedText}\n")
        summaryFile.write(f"Extracted: {extractedText}\n")

def logBatchResult(batchResults):
    """Log batch processing results to file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Write detailed batch log
    with open("batch-log.txt", "a") as logFile:
        logFile.write(f"\n{'#'*60}\n")
        logFile.write(f"BATCH RUN - {timestamp}\n")
        logFile.write(f"{'#'*60}\n")
        
        totalAccuracy = 0
        for result in batchResults:
            logFile.write(f"\nImage: {result['image']}\n")
            logFile.write(f"  Expected: {result['expected']}\n")
            logFile.write(f"  Extracted: {result['extracted']}\n")
            logFile.write(f"  Accuracy: {result['accuracy']}%\n")
            logFile.write(f"  Time: {result['time']:.2f}s\n")
            totalAccuracy += result['accuracy']
        
        avgAccuracy = totalAccuracy / len(batchResults) if batchResults else 0
        logFile.write(f"\n{'='*60}\n")
        logFile.write(f"AVERAGE ACCURACY: {avgAccuracy:.2f}%\n")
        logFile.write(f"TOTAL IMAGES: {len(batchResults)}\n")
        logFile.write(f"{'='*60}\n")
    
    # Write batch summary
    with open("batch-summary.txt", "w") as summaryFile:
        summaryFile.write(f"Batch Run - {timestamp}\n")
        summaryFile.write(f"Average Accuracy: {avgAccuracy:.2f}%\n")
        summaryFile.write(f"Total Images: {len(batchResults)}\n")
        summaryFile.write(f"\nIndividual Results:\n")
        for result in batchResults:
            summaryFile.write(f"  {result['image']}: {result['accuracy']}%\n")

def preprocessImage(image, denoise=False, sharpen=False):
    """Apply image preprocessing to improve OCR accuracy"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Resize if image is too small (minimum height 300px)
    if h < 300:
        scale = 300 / h
        newW = int(w * scale)
        gray = cv2.resize(gray, (newW, 300), interpolation=cv2.INTER_CUBIC)
    
    # Apply adaptive thresholding for better text detection
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    result = binary
    
    # Optional noise reduction
    if denoise:
        result = cv2.medianBlur(result, 3)
    
    # Optional edge sharpening
    if sharpen:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        result = cv2.filter2D(result, -1, kernel)
    
    return result

def captureFromWebcam():
    """Capture a single image from webcam. Returns image or None if cancelled."""
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
        if key == 32:  # SPACE key
            capturedImage = frame
            break
        elif key == 27:  # ESC key
            capturedImage = None
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return capturedImage

def extractTextFromImage(image, denoise=False, sharpen=False):
    """Extract text from image (can be file path string or numpy array)"""
    # Load image if path provided, otherwise use direct image
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            return f"Error: Could not read image '{image}'."
    else:
        img = image
    
    # Preprocess the image
    processed = preprocessImage(img, denoise, sharpen)
    
    # Initialize EasyOCR reader (English only)
    reader = easyocr.Reader(['en'], verbose=False, gpu=False)
    
    # Extract text
    results = reader.readtext(processed, detail=0, paragraph=False)
    
    # Join all detected text
    extractedText = " ".join(results)
    
    return extractedText if extractedText else "No text detected"

def detectTestCase(imagePath):
    """Auto-detect which test case matches the image filename"""
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
        return None, None

def findEasyImages():
    """Find all images in the 'easy images' folder"""
    images = []
    
    # Check if folder exists
    if os.path.exists("easy images"):
        # Loop through all files in folder
        for file in os.listdir("easy images"):
            # Filter for image file extensions
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                images.append(os.path.join("easy images", file))
    
    images.sort()
    return images

# ==================== MAIN PROGRAM START ====================

print("=" * 55)
print("IhaveTheVision")
print("=" * 55)

# Mode selection menu
print("\nSelect mode:")
print("  1. Single image")
print("  2. Batch process easy images (test1, test2, test3)")
print("  3. Webcam (take a picture)")

modeChoice = input("\nChoose (1, 2, or 3): ").strip()

# BATCH MODE - Process all images in 'easy images' folder
if modeChoice == "2":
    # Find all images in easy images folder
    easyImages = findEasyImages()
    
    print(f"[INFO] Found {len(easyImages)} images:")
    for img in easyImages:
        print(f"  - {img}")
    
    # Get preprocessing preferences
    print("\nOptional preprocessing? (Choose only one, both gives quack results):")
    denoise = input("Apply denoising? (y/n): ").strip().lower() in ['y', 'yes']
    sharpen = input("Apply sharpening? (y/n): ").strip().lower() in ['y', 'yes']
    
    batchResults = []
    totalStart = time.time()
    
    # Process each image
    for i, imagePath in enumerate(easyImages, 1):
        print("processing: {imagePath}")
        
        # Detect which test case this image belongs to
        testKey, expectedText = detectTestCase(imagePath)
        
        # Extract text
        startTime = time.time()
        result = extractTextFromImage(imagePath, denoise, sharpen)
        elapsed = time.time() - startTime
        
        # Calculate accuracy
        accuracy = calculateAccuracy(result, expectedText)
        
        # Store results
        batchResults.append({
            'image': imagePath,
            'expected': expectedText,
            'extracted': result,
            'accuracy': accuracy,
            'time': elapsed
        })
        
        # Print progress
        print(f"  Expected: {expectedText[:50]}...")
        print(f"  Extracted: {result[:50]}...")
        print(f"  Accuracy: {accuracy}%")
        print(f"  Time: {elapsed:.2f}s")
    
    # Calculate batch statistics
    totalElapsed = time.time() - totalStart
    avgAccuracy = sum(r['accuracy'] for r in batchResults) / len(batchResults) if batchResults else 0
    
    # Print batch summary
    print("\n" + "=" * 55)
    print("BATCH COMPLETE")
    print("=" * 55)
    print(f"Total Images: {len(batchResults)}")
    print(f"Average Accuracy: {avgAccuracy:.2f}%")
    print(f"Total Time: {totalElapsed:.2f} seconds")
    print("=" * 55)
    
    # Log results
    logBatchResult(batchResults)
    print("\n[INFO] Batch results logged to 'batch-log.txt'")

# WEBCAM MODE - Capture and process single image from webcam
elif modeChoice == "3":
    print("\n[INFO] Starting webcam...")
    imageData = captureFromWebcam()
    if imageData is None:
        print("Cancelled or webcam error.")
        sys.exit(0)
    imageIdentifier = "webcam_capture"
    print("\n[INFO] Image captured from webcam")
    
    # Get preprocessing preferences
    print("\nOptional preprocessing (Choose only one, both gives quack results):")
    denoise = input("Apply denoising? (y/n): ").strip().lower() in ['y', 'yes']
    sharpen = input("Apply sharpening? (y/n): ").strip().lower() in ['y', 'yes']
    
    print(f"\nAnalyzing...\n")
    
    # Extract text
    startTime = time.time()
    result = extractTextFromImage(imageData, denoise, sharpen)
    elapsed = time.time() - startTime
    
    # Print results
    print("-" * 55)
    print("EXTRACTED TEXT:")
    print("-" * 55)
    print(result)
    print("-" * 55)
    print(f"\nTime: {elapsed:.2f} seconds")
    
    # Log webcam capture
    with open("latest-log.txt", "a") as logFile:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logFile.write(f"\n{'='*60}\nTimestamp: {timestamp}\nImage: webcam_capture\nExtracted: {result}\nTime: {elapsed:.2f}s\n{'='*60}\n")

# SINGLE IMAGE MODE - Process one specific image
else:
    # Get image filename from user
    imageName = input("\nImage filename: ").strip().strip('"').strip("'")
    imageData = imageName
    imageIdentifier = imageName
    
    # Auto-detect test case from filename
    testKey, expectedText = detectTestCase(imageName)
    if testKey:
        print(f"\n[INFO] Auto-detected test case: {testKey}")
        print(f"[INFO] Expected: {expectedText[:100]}...")
    
    # Get preprocessing preferences
    print("\nOptional preprocessing? (Choose only one, both gives quack results):")
    denoise = input("Apply denoising? (y/n): ").strip().lower() in ['y', 'yes']
    sharpen = input("Apply sharpening? (y/n): ").strip().lower() in ['y', 'yes']
    
    print(f"\nAnalyzing...\n")
    
    # Extract text
    startTime = time.time()
    result = extractTextFromImage(imageData, denoise, sharpen)
    elapsed = time.time() - startTime
    
    # Print results
    print("-" * 55)
    print("EXTRACTED TEXT:")
    print("-" * 55)
    print(result)
    print("-" * 55)
    print(f"\nTime: {elapsed:.2f} seconds")
    
    # If test case detected, calculate and log accuracy
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
    else:
        # Log without accuracy calculation
        with open("latest-log.txt", "a") as logFile:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logFile.write(f"\n{'='*60}\nTimestamp: {timestamp}\nImage: {imageIdentifier}\nExtracted: {result}\nTime: {elapsed:.2f}s\n{'='*60}\n")
        print("\n[INFO] No test case detected for this filename (looking for 'test1', 'test2', or 'test3')")