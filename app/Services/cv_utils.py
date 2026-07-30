import cv2

face_cascade = cv2.CascadeClassifier("app/models/haarcascade_frontalface_default.xml")

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def resize(img, w=224, h=224):
    return cv2.resize(img, (w, h))

def blur(img, k=5):
    return cv2.GaussianBlur(img, (k, k), 0)

def edges(img):
    return cv2.Canny(to_gray(img), 100, 200)

def detect_faces(img):
    gray = to_gray(img)
    boxes = face_cascade.detectMultiScale(gray, 1.1, 5)
    return boxes  # list of (x, y, w, h)

def draw_boxes(img, boxes):
    out = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out

if __name__ == "__main__":
    img = cv2.imread("data/sample.jpg")
    boxes = detect_faces(img)
    cv2.imwrite("data/sample_faces.jpg", draw_boxes(img, boxes))
    print(f"Found {len(boxes)} face(s)")
