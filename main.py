import os
import time
import traceback
import pyautogui
from PIL import ImageGrab
import pytesseract
import cv2
import numpy as np
from ollama import chat

# Config

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pyautogui.PAUSE = 0.1
pyautogui.FAILSAFE = True

SAVE_DIR = os.path.join(os.path.expanduser("~"), "Desktop")
os.makedirs(SAVE_DIR, exist_ok=True)

CAPTURE_PATH = os.path.join(SAVE_DIR, "captured_screen.png")
PROCESSED_PATH = os.path.join(SAVE_DIR, "processed_image.png")

SCREEN_REGION = (191, 252, 1703, 982)

# Main logic
try:
    print("Starting in 3 seconds... Make sure the question is visible within the capture box!")
    time.sleep(3)

    print(f"Screen: {SCREEN_REGION}")
    img = ImageGrab.grab(bbox=SCREEN_REGION)
    img.save(CAPTURE_PATH)
    print(f"Saved: '{CAPTURE_PATH}'")

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(PROCESSED_PATH, th)
    print(f"Saved image: '{PROCESSED_PATH}'")

    print("\n Text...")
    text = pytesseract.image_to_string(th, lang='eng', config='--psm 6 --oem 3')
    print(f"\n{'='*60}\nEXTRACTED TEXT:\n{'='*60}\n{text}\n{'='*60}\n")

    if not text.strip():
        print("No text")
        raise SystemExit

    print("Prossesing")
    response = chat(
        model='llama3.2',
        messages=[{
            'role': 'user',
            'content': f"{text}\n\nWork this out. Do not explain, just give one answer as a number.",
        }]
    )

    answer = ""
    if isinstance(response, dict):
        answer = response.get("message", {}).get("content", "")
    elif hasattr(response, "message"):
        answer = getattr(response.message, "content", "")
    elif hasattr(response, "__iter__"):
        for chunk in response:
            if "message" in chunk and "content" in chunk["message"]:
                answer += chunk["message"]["content"]

    answer = (answer or "").strip()

    if not answer:
        print("No answer")
        raise SystemExit

    print(f"\n{'='*60}\nAI RESPONSE:\n{'='*60}\n{answer}\n{'='*60}\n")

#TODO: Make mouse go to input automatically

    print("5 seconds to click answer box")
    for i in range(5, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    print("Typing")
    pyautogui.write(answer, interval=0.05)
    time.sleep(0.3)
    pyautogui.press("enter")
    print("Enter")

# Error

except PermissionError as e:
    print(f"\n Permission error: {e}")
    print("Try running this script outside of OneDrive or move it to Desktop.")

except Exception as e:
    print(f"\n Unexpected error: {e}")
    traceback.print_exc()
