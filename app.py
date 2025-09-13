

import streamlit as st
import sys
import os
import base64
from datetime import datetime
# إضافة المسارات للمجلدات


from config import set_page_config, apply_custom_css, get_default_image, apply_animations
from helpers import load_image, save_image, get_image_info, add_achievement
from achievements import init_achievements, check_achievements, display_achievements

# إعداد الصفحة
set_page_config()

# تطبيق التنسيقات المخصصة والحركات
apply_custom_css()
apply_animations()
# استيراد الوحدات
from module1_intro import show_module1
from module2_colors import show_module2
from module3_operations import show_module3
from module4_filters import show_module4
from module5_denoising import show_module5
from module6_edges import show_module6
from module7_morphological import show_module7
from module8_geometric import show_module8
from module9_final import show_module9
from quiz import show_quiz
from challenges import show_challenges

# تهيئة حالة الجلسة
if 'user_name' not in st.session_state:
    st.session_state.user_name = None
if 'progress' not in st.session_state:
    st.session_state.progress = {f"module{i}": False for i in range(1, 10)}
if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = 0
if 'achievements' not in st.session_state:
    st.session_state.achievements = []
if 'user_level' not in st.session_state:
    st.session_state.user_level = 1
if 'user_xp' not in st.session_state:
    st.session_state.user_xp = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.now()

# تهيئة الإنجازات
init_achievements()

# شريط جانبي للتنقل بين المحاضرات
st.sidebar.title("🎓 سلسلة محاضرات معالجة الصور")
st.sidebar.markdown("---")

# معلومات المستخدم
if st.session_state.user_name is None:
    with st.sidebar.form("user_info"):
        st.subheader("معلومات المتعلم")
        name = st.text_input("أدخل اسمك")
        email = st.text_input("البريد الإلكتروني (اختياري)")
        submitted = st.form_submit_button("بدء التعلم")
        if submitted and name:
            st.session_state.user_name = name
            st.session_state.user_email = email
            st.session_state.start_time = datetime.now()
            add_achievement("المبتدئ", "بدء أول محاضرة")
            st.rerun()

if st.session_state.user_name:
    st.sidebar.success(f"مرحباً، {st.session_state.user_name}!")
    
    # عرض تقدم المستخدم
    progress_count = sum(st.session_state.progress.values())
    st.sidebar.progress(progress_count / 9)
    st.sidebar.caption(f"التقدم: {progress_count}/9 محاضرات")
    
    # عرض المستوى والخبرة
    st.sidebar.markdown(f"**المستوى:** {st.session_state.user_level}")
    st.sidebar.markdown(f"**النقاط:** {st.session_state.user_xp}")
    
    # قائمة المحاضرات
    st.sidebar.markdown("### 📚 المحاضرات")
    modules = {
        "المحاضرة 1: مدخل ومعمارية الصور الرقمية": show_module1,
        "المحاضرة 2: أنظمة الألوان": show_module2,
        "المحاضرة 3: العمليات على البكسل": show_module3,
        "المحاضرة 4: الفلاتر والالتفاف": show_module4,
        "المحاضرة 5: إزالة الضوضاء": show_module5,
        "المحاضرة 6: كشف الحواف": show_module6,
        "المحاضرة 7: العمليات المورفولوجية": show_module7,
        "المحاضرة 8: التحويلات الهندسية": show_module8,
        "المحاضرة 9: المشروع الختامي": show_module9,
        "التحديات العملية": show_challenges,
        "الاختبار النهائي": show_quiz
    }
    
    selected_module = st.sidebar.radio("اختر المحاضرة:", list(modules.keys()))
    
    # زر الإنجازات
    if st.sidebar.button("🏆 الإنجازات"):
        display_achievements()
    
    # زر المساعدة
    if st.sidebar.button("❓ المساعدة"):
        st.sidebar.info("""
        - اختر محاضرة من القائمة لبدء التعلم
        - كل محاضرة تحتوي على شرح نظري وتطبيق عملي
        - يمكنك رفع صورك أو استخدام الصور الافتراضية
        - احصل على نقاط الخبرة عند إكمال المحاضرات
        """)
    
    # عرض المحتوى
    st.sidebar.markdown("---")
    modules[selected_module]()
    
    # التحقق من الإنجازات
    check_achievements()
else:
    # صفحة الترحيب
    st.title("🎓 سلسلة محاضرات معالجة الصور التفاعلية")

    # شريط فيديو توضيحي (يمكن استبداله بصورة)
    st.video("https://www.youtube.com/watch?v=oXlwWbU8l2o", format="video/mp4", start_time=0)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## مرحباً بك في دورة معالجة الصور التفاعلية!
        
        هذه الدورة مصممة لمساعدتك على فهم أساسيات معالجة الصور الرقمية بطريقة تفاعلية وسهلة.
        
        ### 🎯 ما الذي ستتعلمه؟
        - أساسيات الصور الرقمية وأنظمة الألوان
        - العمليات الأساسية على البكسلات
        - تطبيق الفلاتر والتحويلات المختلفة
        - كشف الحواف وإزالة الضوضاء
        - العمليات المورفولوجية والتحويلات الهندسية
        
        ### 🚀 ميزات المنصة:
        - واجهة تفاعلية باللغة العربية
        - شرح نظري واضح مع أمثلة عملية
        - نظام نقاط وإنجازات
        - اختبارات وتحديات عملية
        - إمكانية رفع الصور الخاصة بك
        
        **لبدء رحلة التعلم، يرجى إدخال اسمك في الشريط الجانبي.**
        """)
    
    with col2:
        st.image("https://via.placeholder.com/300x400/4CAF50/FFFFFF?text=معالجة+الصور", use_container_width=True)
        st.markdown("""
        ### 📊 إحصائيات الدورة:
        - 9 محاضرات شاملة
        - 15+ أداة تفاعلية
        - 10+ تحديات عملية
        - شهادة إكمال إلكترونية
        """)
    
    # عرض معاينة للصور
    st.markdown("## 📸 معاينة للتجارب العملية")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("https://via.placeholder.com/300x200/2196F3/FFFFFF?text=الصورة+الأصلية", caption="الصورة الأصلية", use_container_width=True)
    with col2:
        st.image("https://via.placeholder.com/300x200/FF5722/FFFFFF?text=كشف+الحواف", caption="كشف الحواف", use_container_width=True)
    with col3:
        st.image("https://via.placeholder.com/300x200/4CAF50/FFFFFF?text=تطبيق+فلتر", caption="تطبيق فلتر", use_container_width=True)
    
    # آراء المستخدمين
    st.markdown("## 💬 آراء المتعلمين")
    testimonials = [
        {"name": "أحمد", "comment": "دورة رائعة ساعدتني في فهم أساسيات معالجة الصور بشكل عملي."},
        {"name": "فاطمة", "comment": "الواجهة العربية والشرح الواضح جعلوا التعلم أسهل بكثير."},
        {"name": "محمد", "comment": "نظام الإنجازات والتحديات شجعني على إكمال جميع المحاضرات."}
    ]
    
    for testimonial in testimonials:
        st.info(f"**{testimonial['name']}:** {testimonial['comment']}")

# تذييل الصفحة
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>© 2023 سلسلة محاضرات معالجة الصور التفاعلية | تم التطوير باستخدام Streamlit و OpenCV</p>
    </div>
    """,
    unsafe_allow_html=True
)