import streamlit as st
import cv2
import numpy as np
from helpers import load_image, add_user_progress, add_achievement

def show_module2():
    """عرض المحاضرة الثانية: أنظمة الألوان"""
    
    st.header("🎨 المحاضرة 2: أنظمة الألوان (Color Spaces)")
    
    # معلومات التقدم
    if st.session_state.progress.get("module2", False):
        st.success("✅ لقد أكملت هذه المحاضرة بالفعل")
    else:
        if add_user_progress("module2"):
            st.balloons()
            st.success("🎉 مبروك! لقد أكملت المحاضرة الثانية وحصلت على 20 نقطة")
            add_achievement("خبير الألوان", "إكمال وحدة أنظمة الألوان")
    
    # النظرية
    with st.expander("📖 الشرح النظري", expanded=True):
        st.markdown("""
        ## أنظمة الألوان المختلفة وأهميتها في معالجة الصور

        ### نظام RGB (Red, Green, Blue)
        - **الاستخدام**: النظام الأكثر شيوعاً في الشاشات والعرض
        - **المميزات**: 
          * يتكون من 3 قنوات (أحمر، أخضر، أزرق)
          * كل قناة تمثل بقيمة بين 0-255
          * الألوان تتكون بجمع هذه القنوات
        
        ### نظام BGR (Blue, Green, Red)
        - **الاستخدام**: النظام الافتراضي في OpenCV
        - **الفرق عن RGB**: ترتيب القنوات معكوس
        
        ### نظام Grayscale (التدرج الرمادي)
        - **الاستخدام**: عندما لا نحتاج لمعلومات الألوان
        - **المميزات**:
          * حجم بيانات أصغر
          * أسرع في المعالجة
          * مناسب للكثير من خوارزميات الرؤية الحاسوبية
        
        ### نظام HSV (Hue, Saturation, Value)
        - **Hue (الصبغة)**: نوع اللون (0-180 في OpenCV)
        - **Saturation (الإشباع)**: نقاء اللون (0-255)
        - **Value (القيمة)**: سطوع اللون (0-255)
        - **الاستخدام**: ممتاز لفصل الألوان based on color ranges
        
        ### نظام YCrCb
        - **Y**: component الإضاءة (Luma)
        - **Cr**: difference between red and luma
        - **Cb**: difference between blue and luma
        - **الاستخدام**: ضغط الصور (JPEG)
        
        ### نظام LAB
        - **L**: الإضاءة
        - **A**: من الأخضر إلى الأحمر
        - **B**: من الأزرق إلى الأصفر
        - **الاستخدام**: قياس الفروق بين الألوان بشكل دقيق
        """)
    
    st.markdown("---")
    
    # التطبيق العملي
    st.subheader("🔍 التجربة العملية: التحويل بين أنظمة الألوان")
    
    # تحميل الصورة
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📤 تحميل الصورة")
        uploaded_file = st.file_uploader("اختر صورة ملونة", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = load_image(uploaded_file)
            # التأكد أن الصورة ملونة
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            st.image(image, caption="الصورة الأصلية (RGB)", use_container_width=True)
        else:
            # إنشاء صورة افتراضية ملونة
            image = np.zeros((300, 400, 3), dtype=np.uint8)
            cv2.rectangle(image, (50, 50), (150, 150), (0, 0, 255), -1)  # أحمر
            cv2.rectangle(image, (150, 50), (250, 150), (0, 255, 0), -1)  # أخضر
            cv2.rectangle(image, (250, 50), (350, 150), (255, 0, 0), -1)  # أزرق
            cv2.circle(image, (200, 200), 50, (255, 255, 0), -1)  # أزرق+أخضر
            cv2.circle(image, (100, 200), 50, (255, 0, 255), -1)  # أزرق+أحمر
            cv2.circle(image, (300, 200), 50, (0, 255, 255), -1)  # أخضر+أحمر
            st.image(image, caption="الصورة الافتراضية الملونة", use_container_width=True)
    
    with col2:
        if image is not None:
            st.markdown("#### 🎛️ اختر نظام الألوان للتحويل")
            
            color_space = st.selectbox(
                "نظام الألوان:",
                ["GRAY", "HSV", "LAB", "YCrCb", "BGR"]
            )
            
            if st.button("🔃 تطبيق التحويل"):
                with st.spinner("جاري التحويل..."):
                    converted_image = convert_color_space(image, color_space)
                    
                    if converted_image is not None:
                        st.image(converted_image, 
                                caption=f"الصورة المحولة إلى {color_space}", 
                                use_container_width=True)
                        
                        # عرض معلومات إضافية
                        st.info(f"**معلومات الصورة المحولة:**")
                        st.write(f"الأبعاد: {converted_image.shape}")
                        st.write(f"نوع البيانات: {converted_image.dtype}")
                        st.write(f"عدد القنوات: {converted_image.shape[2] if len(converted_image.shape) > 2 else 1}")
    
    st.markdown("---")
    
    # قسم إضافي: تقسيم القنوات
    st.subheader("🔬 تقسيم القنوات اللونية")
    
    if image is not None:
        st.markdown("##### قنوات RGB المنفصلة:")
        
        # تقسيم القنوات
        if len(image.shape) == 3 and image.shape[2] == 3:
            b, g, r = cv2.split(image)
            
            cols = st.columns(3)
            channels = [("🔴 الأحمر", r), ("🟢 الأخضر", g), ("🔵 الأزرق", b)]
            
            for i, (name, channel) in enumerate(channels):
                with cols[i]:
                    # إنشاء صورة للقناة مع تلوينها
                    channel_display = np.zeros_like(image)
                    channel_display[:,:,i] = channel
                    st.image(channel_display, caption=name, use_container_width=True)
                    st.metric(f"متوسط {name.split()[-1]}", f"{np.mean(channel):.1f}")
        
        # تطبيق عملي للHSV
        st.markdown("##### تطبيق عملي: فصل الألوان في مساحة HSV")
        
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(h, caption="Hue (الصبغة)", use_container_width=True, clamp=True)
        with col2:
            st.image(s, caption="Saturation (الإشباع)", use_container_width=True, clamp=True)
        with col3:
            st.image(v, caption="Value (القيمة)", use_container_width=True, clamp=True)
        
        # أداة لفصل لون محدد
        st.markdown("##### 🎯 أداة فصل لون محدد في HSV")
        
        col1, col2 = st.columns(2)
        with col1:
            hue_min = st.slider("Hue الأدنى", 0, 180, 0)
            hue_max = st.slider("Hue الأعلى", 0, 180, 180)
        with col2:
            sat_min = st.slider("Saturation الأدنى", 0, 255, 0)
            sat_max = st.slider("Saturation الأعلى", 0, 255, 255)
        
        # إنشاء mask للون المحدد
        lower_bound = np.array([hue_min, sat_min, 0])
        upper_bound = np.array([hue_max, sat_max, 255])
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        result = cv2.bitwise_and(image, image, mask=mask)
        
        st.image(result, caption="المنطقة الملونة المحددة", use_container_width=True)
    
    # اختبار قصير
    st.markdown("---")
    with st.expander("🧪 اختبار قصير", expanded=False):
        st.subheader("اختبار فهم أنظمة الألوان")
        
        q1 = st.radio(
            "1. ما هو النظام اللوني الأفضل لفصل الألوان based on color ranges?",
            ["RGB", "HSV", "GRAY", "BGR"]
        )
        
        q2 = st.radio(
            "2. كم قناة في نظام HSV?",
            ["1", "2", "3", "4"]
        )
        
        q3 = st.radio(
            "3. أي نظام ألوان يستخدمه OpenCV افتراضياً?",
            ["RGB", "BGR", "HSV", "LAB"]
        )
        
        if st.button("✅ التحقق من الإجابات"):
            score = 0
            if q1 == "HSV":
                score += 1
                st.success("السؤال 1: صحيح! HSV ممتاز لفصل الألوان")
            else:
                st.error("السؤال 1: خطأ! HSV هو الأفضل لفصل الألوان")
            
            if q2 == "3":
                score += 1
                st.success("السؤال 2: صحيح! HSV يحتوي على 3 قنوات")
            else:
                st.error("السؤال 2: خطأ! HSV يحتوي على 3 قنوات (H, S, V)")
            
            if q3 == "BGR":
                score += 1
                st.success("السؤال 3: صحيح! OpenCV يستخدم BGR افتراضياً")
            else:
                st.error("السؤال 3: خطأ! OpenCV يستخدم BGR افتراضياً")
            
            st.info(f"**النتيجة: {score}/3**")
            
            if score == 3:
                st.balloons()
                st.session_state.quiz_score = st.session_state.get('quiz_score', 0) + 3

def convert_color_space(image, color_space):
    """تحويل مساحة الألوان للصورة"""
    try:
        if color_space == "GRAY":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif color_space == "HSV":
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif color_space == "LAB":
            return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif color_space == "YCrCb":
            return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        elif color_space == "BGR":
            return image  # نفس الصورة
        else:
            return image
    except Exception as e:
        st.error(f"خطأ في التحويل: {e}")
        return None