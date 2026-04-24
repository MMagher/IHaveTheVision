import os
import sys
import time
import re
from datetime import datetime

# Import required libraries with error handling
try:
    import cv2
    import easyocr
    import numpy as np
except ImportError as e:
    # Extract the missing package name from error message
    missing = str(e).split("'")[1] if "'" in str(e) else "unknown"
    print(f"Missing: {missing}. Run: pip install opencv-python easyocr numpy")
    sys.exit(1)

# Test cases dictionary - maps test IDs to expected text
testcase = {
    "e1": "PASSPORT",
    "e2": "git push -u origin main",
    "e3": "All exams times are scheduled in EST (eastern Standard Time) Semester Course Title CRN Instructor Date Start Duration Room Location Surname Winter CSCI Artificail 72751 Davoudi April 22 3:30 p.m. 2 hours 15 minutes SIR2060 North 2026 4610U Intelligence",
    "e4": "7C Hamilton 8C 3C Mostly Sunny",
    "e5": "SUNO"
}

def log(msg):
    """Write a message to the log file"""
    with open("logs/latest-log.txt", "a") as f:
        f.write(msg + "\n")

def logSeparator(char="=", length=60):
    """Write a separator line to the log file"""
    log(char * length)

def accuracy(extracted, expected):
    """Calculate accuracy between extracted and expected text (0-100%)"""
    if not extracted or extracted == "No text detected":
        return 0.0
    
    extracted, expected = str(extracted).strip(), str(expected).strip()
    
    # Perfect match returns 100%
    if extracted == expected:
        return 100.0
    
    maxLen = max(len(extracted), len(expected))
    if maxLen == 0:
        return 0.0
    
    # Character-level accuracy - count matching characters
    charMatches = sum(1 for i in range(min(len(extracted), len(expected))) if extracted[i] == expected[i])
    charAcc = (charMatches / maxLen) * 100
    
    # Word-level accuracy - count matching words
    extractedWords = set(re.findall(r'\b\w+\b', extracted.lower()))
    expectedWords = set(re.findall(r'\b\w+\b', expected.lower()))
    wordAcc = (len(extractedWords & expectedWords) / len(expectedWords)) * 100 if expectedWords else 0
    
    # Combine: 40% character, 60% word accuracy
    return round((charAcc * 0.4) + (wordAcc * 0.6), 2)

def preprocess(img, denoise=False, sharpen=False):
    """Preprocess image to improve OCR accuracy"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Resize small images (minimum height 300px)
    if h < 300:
        scale = 300 / h
        gray = cv2.resize(gray, (int(w * scale), 300), interpolation=cv2.INTER_CUBIC)
    
    # Apply adaptive thresholding for better text detection
    result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Optional noise reduction
    if denoise:
        result = cv2.medianBlur(result, 3)
    
    # Optional edge sharpening
    if sharpen:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        result = cv2.filter2D(result, -1, kernel)
    
    return result

def extract(img, denoise=False, sharpen=False):
    """Extract text from image (file path or numpy array)"""
    # Load image if path is provided
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            return "Error: Could not read image"
    
    # Preprocess the image
    processed = preprocess(img, denoise, sharpen)
    
    # Initialize OCR reader (English only)
    reader = easyocr.Reader(['en'], verbose=False, gpu=False)
    
    # Extract and return text
    return " ".join(reader.readtext(processed, detail=0, paragraph=False)) or "No text detected"

def webcam():
    """Capture an image from webcam. Returns image or None."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return None
    
    print("\n[INFO] Press SPACE to capture, ESC to cancel")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("IhaveTheVision - SPACE to capture", frame)
        key = cv2.waitKey(1)
        if key == 32:  # SPACE key - capture
            cap.release()
            cv2.destroyAllWindows()
            return frame
        elif key == 27:  # ESC key - cancel
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return None

def testCase(path):
    """Auto-detect which test case matches the image filename"""
    f = os.path.basename(path).lower()
    # Check for test1-5 or e1-5 in filename
    for i in range(1, 6):
        if f'test{i}' in f or f'e{i}' in f:
            return f'e{i}', testcase[f'e{i}']
    return None, None

def findImages():
    """Find all images in the 'easy images' folder"""
    if not os.path.exists("easy images"):
        return []
    # Return full paths for common image formats
    return [os.path.join("easy images", f) for f in os.listdir("easy images") 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

def singleMode():
    """Process a single image file"""
    # Get image filename from user
    name = input("\nImage filename: ").strip().strip('"').strip("'")
    key, expected = testCase(name)
    if key:
        print(f"\n[INFO] Test: {key}")
    
    # Get preprocessing preferences
    denoise = input("Denoise? (y/n): ").strip().lower() in ['y', 'yes']
    sharpen = input("Sharpen? (y/n): ").strip().lower() in ['y', 'yes']
    
    # Extract text and measure time
    start = time.time()
    result = extract(name, denoise, sharpen)
    elapsed = time.time() - start
    
    # Display results
    print(f"\n{'─'*55}\nEXTRACTED TEXT:\n{'─'*55}\n{result}\n{'─'*55}\nTime: {elapsed:.2f}s")
    
    # If test case exists, calculate and log accuracy
    if key and expected:
        acc = accuracy(result, expected)
        print(f"\n{'='*55}\nTEST RESULT:\n  Expected: {expected}\n  Accuracy: {acc}%\n{'='*55}")
        
        # Log detailed results
        log(f"\n{'='*60}\nTimestamp: {datetime.now():%Y-%m-%d %H:%M:%S}\nImage: {name}\nTime: {elapsed:.2f}s\nAccuracy: {acc}%\nExpected: {expected}\nExtracted: {result}\n{'='*60}")
    else:
        # Log without accuracy
        log(f"\n{'='*60}\nTimestamp: {datetime.now():%Y-%m-%d %H:%M:%S}\nImage: {name}\nTime: {elapsed:.2f}s\nExtracted: {result}\n{'='*60}")
    
    print(f"\n[INFO] Logged to logs/latest-log.txt")

def batchMode():
    """Process all images in the easy images folder"""
    # Find all images
    images = findImages()
    print(f"\n[INFO] Found {len(images)} images")
    for img in images:
        print(f"  - {img}")
    
    # Get preprocessing preferences
    denoise = input("\nDenoise? (y/n): ").strip().lower() in ['y', 'yes']
    sharpen = input("Sharpen? (y/n): ").strip().lower() in ['y', 'yes']
    
    results = []
    totalStart = time.time()
    
    # Process each image
    for path in images:
        key, expected = testCase(path)
        if not key:
            print(f"\n[SKIP] {path} - no test case")
            continue
        
        print(f"\nProcessing: {path}")
        start = time.time()
        text = extract(path, denoise, sharpen)
        elapsed = time.time() - start
        acc = accuracy(text, expected)
        
        # Store result
        results.append({'path': path, 'expected': expected, 'text': text, 'acc': acc, 'time': elapsed})
        print(f"  Expected: {expected[:50]}...\n  Extracted: {text[:50]}...\n  Accuracy: {acc}%\n  Time: {elapsed:.2f}s")
    
    # Display batch summary if any results
    if results:
        totalTime = time.time() - totalStart
        avgAcc = sum(r['acc'] for r in results) / len(results)
        
        print(f"\n{'='*55}\nBATCH COMPLETE\n{'='*55}\nTotal: {len(results)}\nAvg Accuracy: {avgAcc:.2f}%\nTotal Time: {totalTime:.2f}s\n{'='*55}")
        
        # Log batch results
        log(f"\n{'#'*60}\nBATCH RUN - {datetime.now():%Y-%m-%d %H:%M:%S}\n{'#'*60}")
        for r in results:
            log(f"\nImage: {r['path']}\n  Expected: {r['expected']}\n  Extracted: {r['text']}\n  Accuracy: {r['acc']}%\n  Time: {r['time']:.2f}s")
        log(f"\n{'='*60}\nAVERAGE: {avgAcc:.2f}%\nTOTAL: {len(results)}\n{'='*60}")
        
        print("\n[INFO] Logged to logs/latest-log.txt")
    else:
        print("\n[ERROR] No valid test images")

def webcamMode():
    """Capture and process an image from webcam"""
    print("\n[INFO] Starting webcam...")
    img = webcam()
    if img is None:
        print("Cancelled")
        return
    
    # Get preprocessing preferences
    denoise = input("\nDenoise? (y/n): ").strip().lower() in ['y', 'yes']
    sharpen = input("Sharpen? (y/n): ").strip().lower() in ['y', 'yes']
    
    # Extract text
    start = time.time()
    result = extract(img, denoise, sharpen)
    elapsed = time.time() - start
    
    # Display results
    print(f"\n{'─'*55}\nEXTRACTED TEXT:\n{'─'*55}\n{result}\n{'─'*55}\nTime: {elapsed:.2f}s")
    
    # Log webcam result
    log(f"\n{'='*60}\nTimestamp: {datetime.now():%Y-%m-%d %H:%M:%S}\nImage: webcam\nTime: {elapsed:.2f}s\nExtracted: {result}\n{'='*60}")
    print("\n[INFO] Logged to logs/latest-log.txt")

def main():
    """Main program entry point"""
    print("=" * 55)
    print("IhaveTheVision")
    print("=" * 55)
    print("\n1. Single image\n2. Batch (easy images)\n3. Webcam")
    
    choice = input("\nChoose (1, 2, or 3): ").strip()
    
    # Route to selected mode
    if choice == "1":
        singleMode()
    elif choice == "2":
        batchMode()
    elif choice == "3":
        webcamMode()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)