import cv2

print("1. Searching for camera...")
# Try index 0 first (default)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Could not open camera at Index 0.")
    print("Trying Index 1 (External Camera)...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("CRITICAL ERROR: No camera found. Check connections.")
        exit()

print("2. Camera found! Opening video window...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Camera is connected but sending no video.")
        break

    cv2.imshow('Camera Test (Press q to quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()