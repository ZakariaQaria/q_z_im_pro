import streamlit as st
import cv2
import numpy as np
from helpers import load_image

def show_practical_tests():
    """الإختبارات العملية"""
    
    st.header("🔬 الاختبارات العملية")
    st.markdown("طبق ما تعلمته في سيناريوهات عملية حقيقية.")
    
    tests = [
        {
            "id": 1,
            "title": "تحليل صورة طبية",
            "description": "طبق تقنيات المعالجة على صورة أشعة",
            "image": "medical",
            "points": 30
        },
        {
            "id": 2,
            "title": "معالجة صورة فضائية",
            "description": "حسن جودة صورة قمر صناعي",
            "image": "satellite", 
            "points": 35
        },
        {
            "id": 3,
            "title": "فحص جودة منتج",
            "description": "اكتشف العيوب في صورة منتج صناعي",
            "image": "quality",
            "points": 40
        }
    ]
    
    for test in tests:
        with st.expander(f"🔍 {test['title']} - {test['points']} نقطة"):
            st.write(test['description'])
            
            if st.button(f"بدء الاختبار {test['id']}", key=f"test_{test['id']}"):
                st.session_state.current_test = test['id']
                st.rerun()
    
    if 'current_test' in st.session_state:
        handle_practical_test(st.session_state.current_test)

def handle_practical_test(test_id):
    """معالجة الاختبار العملي"""
    
    test_images = {
        1: create_medical_image(),
        2: create_satellite_image(),
        3: create_quality_image()
    }
    
    image = test_images.get(test_id)
    
    if image is None:
        st.error("الصورة غير متاحة")
        return
    
    st.image(image, caption="الصورة المطلوب معالجتها", use_container_width=True)
    
    if test_id == 1:
        medical_test(image)
    elif test_id == 2:
        satellite_test(image)
    elif test_id == 3:
        quality_test(image)

def medical_test(image):
    """اختبار الصورة الطبية"""
    st.markdown("### 🏥 تحليل صورة أشعة")
    st.write("طبق التقنيات المناسبة لتحسين الصورة الطبية:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        technique = st.selectbox("التقنية:", [
            "تحسين التباين",
            "إزالة الضوضاء", 
            "كشف الحواف",
            "العمليات المورفولوجية"
        ])
    
    with col2:
        strength = st.slider("قوة المعالجة", 1, 10, 5)
    
    if st.button("تطبيق المعالجة"):
        result = apply_medical_processing(image, technique, strength)
        st.image(result, caption="الصورة المعالجة", use_container_width=True)
        
        # تقييم النتيجة
        score = evaluate_medical_result(result)
        if score >= 0.7:
            st.success("🎉 معالجة ناجحة!")
            award_points(30)
        else:
            st.warning("⚠️ حاول تحسين المعالجة")

def apply_medical_processing(image, technique, strength):
    """تطبيق معالجة طبية"""
    if technique == "تحسين التباين":
        return cv2.convertScaleAbs(image, alpha=strength/5, beta=0)
    elif technique == "إزالة الضوضاء":
        return cv2.medianBlur(image, strength*2+1)
    elif technique == "كشف الحواف":
        return cv2.Canny(image, 50, 150)
    else:  # العمليات المورفولوجية
        kernel = np.ones((strength, strength), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def evaluate_medical_result(image):
    """تقييم نتيجة المعالجة الطبية"""
    # معايير التقييم المبسطة
    contrast = np.std(image) / 255
    noise = 1 - (np.mean(cv2.Laplacian(image, cv2.CV_64F).var()) / 1000)
    return (contrast + noise) / 2

def award_points(points):
    """منح نقاط للمستخدم"""
    if 'test_points' not in st.session_state:
        st.session_state.test_points = 0
    st.session_state.test_points += points
    st.session_state.user_xp += points

# دوال إنشاء الصور الاختبارية
def create_medical_image():
    """إنشاء صورة أشعة طبية"""
    image = np.zeros((400, 400), dtype=np.uint8)
    # محاكاة صورة أشعة
    cv2.circle(image, (200, 200), 100, 200, -1)
    cv2.circle(image, (200, 200), 50, 100, -1)
    # إضافة ضوضاء
    noise = np.random.normal(0, 30, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def create_satellite_image():
    """إنشاء صورة قمر صناعي"""
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    # محاكاة صورة فضائية
    cv2.rectangle(image, (100, 100), (300, 300), (0, 100, 0), -1)  # منطقة خضراء
    cv2.circle(image, (200, 200), 50, (0, 0, 200), -1)  # منطقة حمراء
    # إضافة ضوضاء
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def create_quality_image():
    """إنشاء صورة فحص جودة"""
    image = np.ones((400, 400), dtype=np.uint8) * 150  # خلفية رمادية
    # منتج سليم
    cv2.rectangle(image, (100, 100), (300, 300), 200, -1)
    # عيوب
    cv2.circle(image, (150, 150), 10, 100, -1)  # عيب دائري
    cv2.rectangle(image, (250, 250), (270, 270), 100, -1)  # عيب مربع
    return image
def quality_test(image):
    """اختبار فحص جودة المنتج"""
    st.markdown("### 🏭 فحص جودة المنتج")
    st.write("اكتشف العيوب في الصورة الصناعية:")
    
    technique = st.selectbox("طريقة الفحص:", [
        "كشف العيوب بالتفتيش",
        "التحليل بالحواف",
        "المقارنة مع القالب"
    ], key="quality_tech")
    
    if st.button("بدء فحص الجودة"):
        with st.spinner("جاري فحص الجودة..."):
            if technique == "كشف العيوب بالتفتيش":
                # تحويل إلى رمادي وكشف العيوب
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                edges = cv2.Canny(gray, 100, 200)
                defects = np.sum(edges > 0)
                
            elif technique == "التحليل بالحواف":
                # تحليل الحواف للكشف عن العيوب
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                edges = cv2.Canny(gray, 50, 150)
                defects = np.sum(edges > 0)
                
            else:  # المقارنة مع القالب
                # مقارنة مع صورة "مثالية"
                template = create_quality_template()
                if len(image.shape) == 3:
                    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    image_gray = image
                diff = cv2.absdiff(image_gray, template)
                defects = np.sum(diff > 50)
            
            st.image(edges if 'edges' in locals() else diff, 
                    caption="المناطق المشبوهة", use_container_width=True)
            
            if defects > 100:
                st.error(f"⚠️ تم اكتشاف {defects} عيب محتمل!")
                st.success("🎉 فحص الجودة مكتمل - تم اكتشاف العيوب")
            else:
                st.success("✅ المنتج سليم - لا توجد عيوب ظاهرة")
            
            award_points(40)

def create_quality_template():
    """إنشاء قالب منتج مثالي"""
    template = np.ones((400, 400), dtype=np.uint8) * 150
    cv2.rectangle(template, (100, 100), (300, 300), 200, -1)
    return template
def satellite_test(image):
    """اختبار الصورة الفضائية"""
    st.markdown("### 🛰️ معالجة الصورة الفضائية")
    st.write("طبق التقنيات المناسبة لتحسين جودة الصورة الفضائية:")
    
    technique = st.selectbox("التقنية:", [
        "تحسين التباين",
        "كشف الحواف", 
        "تحديد المناطق",
        "تنعيم الصورة"
    ], key="satellite_tech")
    
    if st.button("تطبيق المعالجة الفضائية"):
        result = apply_satellite_processing(image, technique)
        st.image(result, caption="الصورة المحسنة", use_container_width=True)
        st.success("🎉 المعالجة مكتملة!")
        award_points(35)

def apply_satellite_processing(image, technique):
    """تطبيق معالجة الصور الفضائية"""
    if technique == "تحسين التباين":
        return cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    
    elif technique == "كشف الحواف":
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return cv2.Canny(gray, 100, 200)
    
    elif technique == "تحديد المناطق":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    else:  # تنعيم الصورة
        return cv2.GaussianBlur(image, (5, 5), 0)
    



