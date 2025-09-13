import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time

def load_image(uploaded_file, use_default=False, default_path=""):
    """تحميل صورة من ملف مرفوع أو استخدام صورة افتراضية"""
    try:
        if use_default and default_path and os.path.exists(default_path):
            image = Image.open(default_path)
            return np.array(image)
        elif uploaded_file is not None:
            image = Image.open(uploaded_file)
            return np.array(image)
        
        # إذا لم يتم توفير صورة، إنشاء صورة افتراضية
        return create_default_image()
    except Exception as e:
        st.error(f"حدث خطأ في تحميل الصورة: {e}")
        return create_default_image()

def create_default_image():
    """إنشاء صورة افتراضية للعرض"""
    # إنشاء صورة ذات خلفية متدرجة
    width, height = 400, 300
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # إضافة تدرج لوني
    for i in range(width):
        color = int(255 * i / width)
        image[:, i] = [color, color//2, 255-color]
    
    # إضافة أشكال هندسية
    cv2.rectangle(image, (50, 50), (150, 150), (0, 255, 0), -1)
    cv2.circle(image, (300, 100), 50, (255, 0, 0), -1)
    cv2.line(image, (200, 50), (250, 200), (0, 0, 255), 3)
    
    # إضافة نص
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'OpenCV', (100, 250), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return image

def save_image(image, filename):
    """حفظ الصورة"""
    try:
        img = Image.fromarray(image)
        img.save(filename)
        return True
    except Exception as e:
        st.error(f"حدث خطأ في حفظ الصورة: {e}")
        return False

def get_image_info(image):
    """الحصول على معلومات الصورة"""
    if image is not None:
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        dtype = image.dtype
        size_kb = image.nbytes / 1024
        
        info = f"""
        **معلومات الصورة:**
        - الأبعاد: {width} × {height} بكسل
        - القنوات: {channels}
        - نوع البيانات: {dtype}
        - الحجم: {size_kb:.2f} كيلوبايت
        """
        return info
    return "لم يتم تحميل صورة"

def apply_brightness_contrast(image, brightness=0, contrast=0):
    """تطبيق السطوع والتباين على الصورة"""
    if image is None:
        return None
    
    # تحويل السطوع والتباين إلى قيم OpenCV
    brightness = int((brightness - 0) * (255 - (-255)) / (100 - 0) + (-255))
    contrast = int((contrast - 0) * (127 - (-127)) / (100 - 0) + (-127))
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        
        buf = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    else:
        buf = image.copy()
    
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    
    return buf

def add_noise(image, noise_type="gaussian", amount=0.05):
    """إضافة ضوضاء إلى الصورة"""
    if image is None:
        return None
    
    if noise_type == "gaussian":
        row, col, ch = image.shape
        mean = 0
        var = amount * 100
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    elif noise_type == "salt_pepper":
        row, col, ch = image.shape
        s_vs_p = 0.5
        out = np.copy(image)
        
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 255
        
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
    
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    return image

def ensure_color(image):
    """تأكد أن الصورة ملونة (3 قنوات)"""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

def ensure_grayscale(image):
    """تأكد أن الصورة رمادية (قناة واحدة)"""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def add_achievement(title, description):
    """إضافة إنجاز للمستخدم"""
    if 'achievements' not in st.session_state:
        st.session_state.achievements = []
    
    achievement = {
        'title': title,
        'description': description,
        'time': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if achievement not in st.session_state.achievements:
        st.session_state.achievements.append(achievement)
        st.session_state.user_xp = st.session_state.get('user_xp', 0) + 10
        
        # التحقق من ترقية المستوى
        if st.session_state.user_xp >= st.session_state.user_level * 100:
            st.session_state.user_level += 1
            st.balloons()
            st.success(f"🎉 مبروك! لقد وصلت إلى المستوى {st.session_state.user_level}")
        
        return True
    return False

def add_user_progress(module_name):
    """تسجيل تقدم المستخدم"""
    if 'progress' not in st.session_state:
        st.session_state.progress = {}
    
    if module_name not in st.session_state.progress or not st.session_state.progress[module_name]:
        st.session_state.progress[module_name] = True
        st.session_state.user_xp = st.session_state.get('user_xp', 0) + 20
        return True
    return False