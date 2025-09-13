import streamlit as st
import cv2
import numpy as np
from helpers import load_image

def show_challenges():
    """عرض التحديات العملية"""
    
    st.header("🏆 التحديات العملية")
    st.markdown("اختبر مهاراتك في معالجة الصور من خلال هذه التحديات العملية.")
    
    challenges = [
        {
            "id": 1,
            "title": "تحسين صورة مظلمة",
            "description": "صورة مظلمة تحتاج إلى تحسين السطوع والتباين",
            "points": 20,
            "completed": False
        },
        {
            "id": 2, 
            "title": "كشف حواف الصورة",
            "description": "استخدم خوارزمية Canny لاكتشاف حواف الصورة",
            "points": 25,
            "completed": False
        },
        {
            "id": 3,
            "title": "إزالة الضوضاء",
            "description": "أزل الضوضاء من الصورة مع الحفاظ على التفاصيل",
            "points": 30,
            "completed": False
        },
        {
            "id": 4,
            "title": "فصل الألوان",
            "description": "افصل لون محدد من الصورة باستخدام HSV",
            "points": 35,
            "completed": False
        },
        {
            "id": 5,
            "title": "المشروع المتكامل",
            "description": "طبق pipeline كامل لمعالجة الصورة",
            "points": 50,
            "completed": False
        }
    ]
    
    # عرض التحديات
    for challenge in challenges:
        with st.expander(f"🏅 التحدي {challenge['id']}: {challenge['title']} ({challenge['points']} نقطة)", expanded=False):
            st.write(challenge['description'])
            
            if st.button(f"بدء التحدي {challenge['id']}", key=f"challenge_{challenge['id']}"):
                st.session_state.current_challenge = challenge['id']
                st.rerun()
    
    # معالجة التحدي الحالي
    if 'current_challenge' in st.session_state:
        current_id = st.session_state.current_challenge
        current_challenge = next((c for c in challenges if c['id'] == current_id), None)
        
        if current_challenge:
            st.markdown(f"## 🎯 {current_challenge['title']}")
            st.write(current_challenge['description'])
            
            # تحميل الصورة المناسبة للتحدي
            challenge_image = create_challenge_image(current_id)
            st.image(challenge_image, caption="الصورة المطلوب معالجتها", use_container_width=True)
            
            # واجهة المعالجة حسب التحدي
            if current_id == 1:
                handle_challenge1(challenge_image)
            elif current_id == 2:
                handle_challenge2(challenge_image)
            elif current_id == 3:
                handle_challenge3(challenge_image)
            elif current_id == 4:
                handle_challenge4(challenge_image)
            elif current_id == 5:
                handle_challenge5(challenge_image)


def create_challenge_image(challenge_id):
    """إنشاء صورة مناسبة للتحدي"""
    if challenge_id == 1:
        # صورة مظلمة ولكن مو سوداء
        image = np.ones((300, 400), dtype=np.uint8) * 50  # رمادي غامق
        cv2.rectangle(image, (50, 50), (150, 150), 150, -1)
        cv2.circle(image, (300, 100), 50, 200, -1)
        return image
    
    elif challenge_id == 2:
        # صورة واضحة
        image = np.zeros((300, 400), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), 255, -1)
        cv2.circle(image, (300, 100), 50, 200, -1)
        return image
    
    elif challenge_id == 3:
        # صورة مع ضوضاء
        image = np.ones((300, 400), dtype=np.uint8) * 100
        cv2.rectangle(image, (50, 50), (150, 150), 200, -1)
        noise = np.random.normal(0, 50, image.shape).astype(np.uint8)
        return cv2.add(image, noise)
    
    elif challenge_id == 4:
        # صورة ملونة
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), (0, 0, 255), -1)  # أحمر
        cv2.rectangle(image, (200, 50), (300, 150), (0, 255, 0), -1)  # أخضر  
        cv2.rectangle(image, (100, 200), (250, 280), (255, 0, 0), -1)  # أزرق
        return image
    
    else:
        # صورة للتحدي 5
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)
        cv2.circle(image, (300, 100), 50, (0, 255, 0), -1)
        return image

def handle_challenge1(image):
    """معالجة التحدي 1: تحسين صورة مظلمة"""
    st.markdown("### 🛠️ أدوات التحسين")
    
    brightness = st.slider("السطوع", -100, 100, 50)
    contrast = st.slider("التباين", 0.0, 3.0, 1.5, 0.1)
    
    if st.button("تطبيق التحسين"):
        enhanced = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        st.image(enhanced, caption="الصورة المحسنة", use_container_width=True)
        
        # تقييم النتيجة
        if np.mean(enhanced) > 100 and np.std(enhanced) > 40:
            st.success("🎉 نجحت في التحدي! الصورة أصبحت واضحة وجيدة التباين")
            award_points(20)
        else:
            st.warning("⚠️ حاول تحسين الإعدادات أكثر")

def handle_challenge2(image):
    """معالجة التحدي 2: كشف الحواف"""
    st.markdown("### 🛠️ أدوات كشف الحواف")
    
    threshold1 = st.slider("العتبة المنخفضة", 0, 255, 100)
    threshold2 = st.slider("العتبة العالية", 0, 255, 200)
    
    if st.button("كشف الحواف"):
        edges = cv2.Canny(image, threshold1, threshold2)
        st.image(edges, caption="الحواف المكتشفة", use_container_width=True)
        
        # تقييم النتيجة
        edge_pixels = np.sum(edges > 0)
        if edge_pixels > 1000 and edge_pixels < 10000:
            st.success("🎉 نجحت في التحدي! تم كشف الحواف بشكل جيد")
            award_points(25)
        else:
            st.warning("⚠️ ضبط العتبات يحتاج تحسين")

def award_points(points):
    """منح نقاط للمستخدم"""
    if 'challenge_points' not in st.session_state:
        
        
        st.session_state.challenge_points = 0
        st.session_state.challenge_points += points
        st.session_state.user_xp += points
        st.success(f"⏫ لقد ربحت {points} نقطة!")
    
def handle_challenge4(image):
    """معالجة التحدي 4: فصل الألوان"""
    st.markdown("### 🎨 أدوات فصل الألوان")
    
    st.info("هذا التحدي يفصل اللون الأحمر من الصورة")
    
    if st.button("فصل اللون الأحمر"):
        # تحويل إلى HSV لفصل الألوان
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # تعريف مدى اللون الأحمر في HSV
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        
        # إنشاء mask للون الأحمر
        mask = cv2.inRange(hsv_image, lower_red, upper_red)
        
        # تطبيق Mask على الصورة
        result = cv2.bitwise_and(image, image, mask=mask)
        
        st.image(result, caption="اللون الأحمر فقط", use_container_width=True)
        
        # تقييم النتيجة
        red_pixels = np.sum(mask > 0)
        if red_pixels > 500:
            st.success("🎉 نجحت في فصل اللون الأحمر!")
            award_points(35)
        else:
            st.warning("⚠️ لم يتم فصل اللون الأحمر بشكل جيد")

def handle_challenge5(image):
    """معالجة التحدي 5: المشروع المتكامل"""
    st.markdown("### 🚀 معالجة متكاملة")
    
    col1, col2 = st.columns(2)
    
    with col1:
        step1 = st.checkbox("تحويل إلى رمادي", value=True)
        step2 = st.checkbox("إزالة الضوضاء", value=True)
        step3 = st.checkbox("كشف الحواف", value=True)
    
    with col2:
        step4 = st.checkbox("عمليات مورفولوجية", value=False)
        step5 = st.checkbox("تحسين التباين", value=True)
    
    if st.button("تشغيل Pipeline المتكامل"):
        result = image.copy()
        
        if step1 and len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        if step2:
            result = cv2.medianBlur(result, 5)
        
        if step3:
            result = cv2.Canny(result, 100, 200)
        
        if step4 and step3:  # فقط إذا تم كشف الحواف
            kernel = np.ones((3, 3), np.uint8)
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        
        if step5 and not step3:  # فقط إذا ما تم كشف الحواف
            result = cv2.convertScaleAbs(result, alpha=1.2, beta=0)
        
        st.image(result, caption="النتيجة النهائية", use_container_width=True)
        st.success("🎉 تمت المعالجة بنجاح!")
        award_points(50)    

def handle_challenge3(image):
    """معالجة التحدي 3: إزالة الضوضاء"""
    st.markdown("### 🛠️ أدوات إزالة الضوضاء")
    
    denoise_method = st.selectbox(
        "طريقة الإزالة:",
        ["Gaussian Filter", "Median Filter", "Bilateral Filter"]
    )
    
    if st.button("تطبيق إزالة الضوضاء"):
        with st.spinner("جاري إزالة الضوضاء..."):
            if denoise_method == "Gaussian Filter":
                result = cv2.GaussianBlur(image, (5, 5), 0)
            elif denoise_method == "Median Filter":
                result = cv2.medianBlur(image, 5)
            elif denoise_method == "Bilateral Filter":
                result = cv2.bilateralFilter(image, 9, 75, 75)
            
            st.image(result, caption="بعد إزالة الضوضاء", use_container_width=True)
            st.success("🎉 تمت إزالة الضوضاء بنجاح!")
            
            # منح النقاط
            if 'challenge_points' not in st.session_state:
                st.session_state.challenge_points = 0
            st.session_state.challenge_points += 30
            st.session_state.user_xp += 30
            st.success("⏫ لقد ربحت 30 نقطة!")        

def handle_challenge3(image):
    """معالجة التحدي 3: إزالة الضوضاء"""
    st.markdown("### 🛠️ أدوات إزالة الضوضاء")
    
    denoise_method = st.selectbox(
        "طريقة الإزالة:",
        ["Gaussian Filter", "Median Filter", "Bilateral Filter"]
    )
    
    if st.button("تطبيق إزالة الضوضاء"):
        with st.spinner("جاري إزالة الضوضاء..."):
            if denoise_method == "Gaussian Filter":
                result = cv2.GaussianBlur(image, (5, 5), 0)
            elif denoise_method == "Median Filter":
                result = cv2.medianBlur(image, 5)
            elif denoise_method == "Bilateral Filter":
                result = cv2.bilateralFilter(image, 9, 75, 75)
            
            st.image(result, caption="بعد إزالة الضوضاء", use_container_width=True)
            st.success("🎉 تمت إزالة الضوضاء بنجاح!")
            award_points(30)

def handle_challenge4(image):
    """معالجة التحدي 4: فصل الألوان"""
    st.markdown("### 🎨 أدوات فصل الألوان")
    st.info("هذا التحدي يفصل اللون الأحمر من الصورة")
    
    if st.button("فصل اللون الأحمر"):
        # تحويل إلى HSV لفصل الألوان
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # تعريف مدى اللون الأحمر في HSV
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        
        # إنشاء mask للون الأحمر
        mask = cv2.inRange(hsv_image, lower_red, upper_red)
        
        # تطبيق Mask على الصورة
        result = cv2.bitwise_and(image, image, mask=mask)
        
        st.image(result, caption="اللون الأحمر فقط", use_container_width=True)
        st.success("🎉 تم فصل اللون الأحمر بنجاح!")
        award_points(35)

def handle_challenge5(image):
    """معالجة التحدي 5: المشروع المتكامل"""
    st.markdown("### 🚀 معالجة متكاملة")
    
    col1, col2 = st.columns(2)
    
    with col1:
        step1 = st.checkbox("تحويل إلى رمادي", value=True)
        step2 = st.checkbox("إزالة الضوضاء", value=True)
        step3 = st.checkbox("كشف الحواف", value=True)
    
    with col2:
        step4 = st.checkbox("عمليات مورفولوجية", value=False)
        step5 = st.checkbox("تحسين التباين", value=True)
    
    if st.button("تشغيل Pipeline المتكامل"):
        result = image.copy()
        
        if step1 and len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        if step2:
            result = cv2.medianBlur(result, 5)
        
        if step3:
            result = cv2.Canny(result, 100, 200)
        
        if step4 and step3:
            kernel = np.ones((3, 3), np.uint8)
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        
        if step5 and not step3:
            result = cv2.convertScaleAbs(result, alpha=1.2, beta=0)
        
        st.image(result, caption="النتيجة النهائية", use_container_width=True)
        st.success("🎉 تمت المعالجة بنجاح!")
        award_points(50)

def award_points(points):
    """منح نقاط للمستخدم"""
    if 'challenge_points' not in st.session_state:
        st.session_state.challenge_points = 0
    st.session_state.challenge_points += points
    st.session_state.user_xp += points
    st.success(f"⏫ لقد ربحت {points} نقطة!")            