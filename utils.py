import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64

# دالة لتحميل الصورة
def load_image(uploaded_file):
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        return img
    return None

# دالة لتحويل صورة OpenCV إلى صيغة يمكن عرضها في Streamlit
def convert_opencv_img_to_streamlit(opencv_image):
    if len(opencv_image.shape) == 2:  # إذا كانت الصورة رمادية
        return Image.fromarray(opencv_image)
    else:  # إذا كانت ملونة
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(opencv_image)

# دالة لحفظ الصورة وتوفير رابط للتحميل
def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# دالة لإنشاء صورة افتراضية
def create_default_image():
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.putText(image, "صورة افتراضية", (100, 150), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image

# دالة لإضافة ضوضاء للصورة
def add_noise(image, noise_type="gaussian"):
    if noise_type == "gaussian":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss * 255
        return np.clip(noisy, 0, 255).astype(np.uint8)
    elif noise_type == "salt_pepper":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.04
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords[0], coords[1], :] = 255
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords[0], coords[1], :] = 0
        return out