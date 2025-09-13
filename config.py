import streamlit as st
import os
import base64
from io import BytesIO

def set_page_config():
    """إعدادات صفحة Streamlit"""
    st.set_page_config(
        page_title="معالجة الصور التفاعلية",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-username/image-processing-course',
            'Report a bug': "https://github.com/your-username/image-processing-course/issues",
            'About': "سلسلة محاضرات تفاعلية في معالجة الصور باستخدام Streamlit و OpenCV"
        }
    )

def apply_custom_css():
    """تطبيق تنسيقات CSS مخصصة"""
    css_file = os.path.join(os.path.dirname(__file__), 'custom.css')
    if os.path.exists(css_file):
        with open(css_file, encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def apply_animations():
    """تطبيق تأثيرات الحركة"""
    css_file = os.path.join(os.path.dirname(__file__), 'animations.css')
    if os.path.exists(css_file):
        with open(css_file, encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def get_default_image(image_name):
    """الحصول على صورة افتراضية"""
    image_path = os.path.join(os.path.dirname(__file__), f'default_images/{image_name}')
    if os.path.exists(image_path):
        return image_path
    return None

def get_image_download_link(img, filename, text):
    """إنشاء رابط لتحميل الصورة"""
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def add_logo():
    """إضافة شعار للتطبيق"""
    logo_path = os.path.join(os.path.dirname(__file__), 'logo.png')
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        st.sidebar.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{data}" width="100">
            </div>
            """,
            unsafe_allow_html=True,
        )