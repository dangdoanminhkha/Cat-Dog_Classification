import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load mô hình đã lưu
model = load_model("dog_cat_classifier.keras")


# Hàm dự đoán loại động vật trong ảnh
def predict_image(image_path):

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Không thể đọc ảnh: {image_path}")
        return

    # Resize ảnh về kích thước phù hợp với mô hình (150x150)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0  # Chuẩn hóa về [0,1]
    img = np.expand_dims(img, axis=0)

    # Thực hiện dự đoán
    prediction = model.predict(img)

    # Xác định kết quả
    label = "Chó 🐶" if prediction[0] > 0.5 else "Mèo 🐱"
    print(f"🔍 Kết quả nhận diện: {label}")


# Chạy thử nghiệm với ảnh đầu vào
image_path = "dog3.jpg"
predict_image(image_path)
