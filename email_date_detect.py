import easyocr
import cv2
import re
import matplotlib.pyplot as plt

# Load the image
image_path = r'D:\Internship\deep_learning\CNN\opencv\data\date_email.jpeg'  # <-- Make sure this path is correct
img = cv2.imread(image_path)

if img is None:
    print("❌ Image not found.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold (optional)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# OCR on thresholded image
results = reader.readtext(thresh)

# Define relaxed regex patterns
email_pattern = r'\b[\w\.-]+\s*@\s*[\w\.-]+\s*[.\s]\s*[a-zA-Z]{2,}\b'
date_pattern = r'\b\d{2}[/-]\d{2}[/-]\d{4}\b'

# Initialize result holders
emails_found = []
dates_found = []

# Scan OCR results
for (bbox, text, confidence) in results:
    if confidence < 0.2:
        continue

    # Clean text
    clean_text = text.strip()
    full_text = clean_text.replace(" ", "").replace("..", ".")

    # Match email
    if re.search(email_pattern, clean_text):
        email_match = re.search(email_pattern, clean_text)
        emails_found.append(email_match.group().replace(" ", "").replace("..", "."))

        # Draw green box for email
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(img, 'Email', top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Match date
    if re.search(date_pattern, clean_text):
        date_match = re.search(date_pattern, clean_text)
        dates_found.append(date_match.group())

        # Draw blue box for date
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)
        cv2.putText(img, 'Date', top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Display found emails and dates
print("📧 Emails found:", emails_found)
print("📅 Dates found:", dates_found)

# Display image with boxes
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 8))
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Detected Emails and Dates")
plt.show()