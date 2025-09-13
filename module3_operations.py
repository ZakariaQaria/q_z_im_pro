import streamlit as st
import cv2
import numpy as np
from helpers import load_image, add_user_progress, apply_brightness_contrast

def show_module3():
    """عرض المحاضرة الثالثة: العمليات على البكسل"""
    
    st.header("✨ المحاضرة 3: العمليات على البكسل (Point Operations)")
    
    # معلومات التقدم
    if st.session_state.progress.get("module3", False):
        st.success("✅ لقد أكملت هذه المحاضرة بالفعل")
    else:
        if add_user_progress("module3"):
            st.balloons()
            st.success("🎉 مبروك! لقد أكملت المحاضرة الثالثة وحصلت على 20 نقطة")
    
    # النظرية
    with st.expander("📖 الشرح النظري", expanded=True):
        st.markdown("""
        ## العمليات على مستوى البكسل (Point Operations)

        العمليات على مستوى البكسل هي transformations تُطبق على كل بكسل independently دون الاعتماد على الجيران.

        ### 1. تعديل السطوع (Brightness Adjustment)
        - **المبدأ**: إضافة أو طرح قيمة ثابتة من جميع وحدات البكسل
        - **الصيغة الرياضية**: `new_pixel = pixel + value`
        - **التأثير**: يجعل الصورة أفتح أو أغمق

        ### 2. تعديل التباين (Contrast Adjustment)
        - **المبدأ**: ضرب قيم البكسلات بعامل تضخيم
        - **الصيغة الرياضية**: `new_pixel = pixel * factor`
        - **التأثير**: يزيد أو يقلل الفروق بين الألوان

        ### 3. الصورة السالبة (Image Negative)
        - **المبدأ**: عكس قيم البكسلات
        - **الصيغة الرياضية**: `new_pixel = 255 - pixel`
        - **التأثير**: يحول الصورة إلى negative مثل الصور الفوتوغرافية القديمة

        ### 4. العتبة (Thresholding)
        - **المبدأ**: تحويل الصورة إلى ثنائية (أبيض وأسود) based on threshold value
        - **الصيغة**: `pixel = 255 if pixel > threshold else 0`
        - **الاستخدام**: فصل objects عن الخلفية

        ### 5. القص (Clipping)
        - **المبدأ**: تحديد نطاق لقيم البكسلات
        - **الصيغة**: `pixel = max(min_value, min(pixel, max_value))`
        - **الاستخدام**: منع overflow أو underflow

        ### 6. التمدد الخطي (Linear Stretching)
        - **المبدأ**: تحسين التباين by stretching intensity range
        - **الصيغة**: `new_pixel = (pixel - min) * (255/(max-min))`
        - **الاستخدام**: تحسين جودة الصور منخفضة التباين

        ### 7. Gamma Correction
        - **المبدأ**: تعديل non-linear للشدة
        - **الصيغة**: `new_pixel = 255 * (pixel/255)^gamma`
        - **الاستخدام**: تصحيح إضاءة الصور
        """)
    
    st.markdown("---")
    
    # التطبيق العملي
    st.subheader("🔧 التجربة العملية: تطبيق العمليات على البكسل")
    
    # تحميل الصورة
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📤 تحميل الصورة")
        uploaded_file = st.file_uploader("اختر صورة", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = load_image(uploaded_file)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            st.image(image, caption="الصورة الأصلية (رمادية)", use_container_width=True)
        else:
            # إنشاء صورة افتراضية
            image = create_sample_grayscale_image()
            st.image(image, caption="الصورة الافتراضية الرمادية", use_container_width=True)
    
    with col2:
        if image is not None:
            st.markdown("#### ⚙️ اختر العملية لتطبيقها")
            
            operation = st.selectbox(
                "نوع العملية:",
                ["السطوع والتباين", "الصورة السالبة", "العتبة الثابتة", "عتبة Otsu", "Gamma Correction"]
            )
            
            if operation == "السطوع والتباين":
                brightness = st.slider("السطوع", -100, 100, 0)
                contrast = st.slider("التباين", -100, 100, 0)
                
                if st.button("🔄 تطبيق السطوع والتباين"):
                    result = apply_brightness_contrast(image, brightness, contrast)
                    st.image(result, caption="بعد تعديل السطوع والتباين", use_container_width=True)
            
            elif operation == "الصورة السالبة":
                if st.button("🔄 تطبيق الصورة السالبة"):
                    result = 255 - image
                    st.image(result, caption="الصورة السالبة", use_container_width=True)
            
            elif operation == "العتبة الثابتة":
                threshold = st.slider("قيمة العتبة", 0, 255, 127)
                
                if st.button("🔄 تطبيق العتبة"):
                    _, result = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
                    st.image(result, caption="بعد تطبيق العتبة", use_container_width=True)
            
            elif operation == "عتبة Otsu":
                if st.button("🔄 تطبيق عتبة Otsu"):
                    _, result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    st.image(result, caption="بعد تطبيق عتبة Otsu", use_container_width=True)
            
            elif operation == "Gamma Correction":
                gamma = st.slider("قيمة Gamma", 0.1, 5.0, 1.0, 0.1)
                
                if st.button("🔄 تطبيق Gamma Correction"):
                    result = apply_gamma_correction(image, gamma)
                    st.image(result, caption=f"بعد Gamma Correction (γ={gamma})", use_container_width=True)
    
    st.markdown("---")
    
    # قسم إضافي: مقارنة بين العمليات
    st.subheader("📊 مقارنة بين تأثير العمليات المختلفة")
    
    if image is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # السطوع العالي
            bright = apply_brightness_contrast(image, 50, 0)
            st.image(bright, caption="سطوع عالي (+50)", use_container_width=True)
        
        with col2:
            # التباين العالي
            contrast_high = apply_brightness_contrast(image, 0, 50)
            st.image(contrast_high, caption="تباين عالي (+50)", use_container_width=True)
        
        with col3:
            # الصورة السالبة
            negative = 255 - image
            st.image(negative, caption="الصورة السالبة", use_container_width=True)
        
        # histogram للصورة
        st.markdown("##### 📈 Histogram للصورة")
        col1, col2 = st.columns(2)
        
        with col1:
            hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])
            st.bar_chart(hist_original)
            st.caption("Histogram للصورة الأصلية")
        
        with col2:
            if 'result' in locals():
                hist_result = cv2.calcHist([result], [0], None, [256], [0, 256])
                st.bar_chart(hist_result)
                st.caption("Histogram بعد المعالجة")
    
    # اختبار قصير
    st.markdown("---")
    with st.expander("🧪 اختبار قصير", expanded=False):
        st.subheader("اختبار فهم العمليات على البكسل")
        
        q1 = st.radio(
            "1. ما تأثير زيادة السطوع على الصورة?",
            ["تزيد التباين", "تجعل الصورة أفتح", "تعكس الألوان", "تقلل الضوضاء"]
        )
        
        q2 = st.radio(
            "2. ما هي الصيغة الرياضية للصورة السالبة?",
            ["pixel * 2", "255 - pixel", "pixel + 100", "pixel / 2"]
        )
        
        q3 = st.radio(
            "3. متى نستخدم عتبة Otsu?",
            ["عندما نعرف قيمة العتبة المثلى", "عندما نريد تحديد العتبة تلقائياً", "لتحسين الألوان", "لزيادة السطوع"]
        )
        
        if st.button("✅ التحقق من الإجابات"):
            score = 0
            if q1 == "تجعل الصورة أفتح":
                score += 1
                st.success("السؤال 1: صحيح! زيادة السطوع تجعل الصورة أفتح")
            else:
                st.error("السؤال 1: خطأ! زيادة السطوع تجعل الصورة أفتح")
            
            if q2 == "255 - pixel":
                score += 1
                st.success("السؤال 2: صحيح! الصورة السالبة = 255 - pixel")
            else:
                st.error("السؤال 2: خطأ! الصيغة الصحيحة هي 255 - pixel")
            
            if q3 == "عندما نريد تحديد العتبة تلقائياً":
                score += 1
                st.success("السؤال 3: صحيح! Otsu يحدد العتبة المثلى تلقائياً")
            else:
                st.error("السؤال 3: خطأ! Otsu يستخدم لتحديد العتبة تلقائياً")
            
            st.info(f"**النتيجة: {score}/3**")
            
            if score == 3:
                st.balloons()
                st.session_state.quiz_score = st.session_state.get('quiz_score', 0) + 3

def create_sample_grayscale_image():
    """إنشاء صورة رمادية افتراضية"""
    image = np.zeros((300, 400), dtype=np.uint8)
    
    # إضافة تدرج رمادي
    for i in range(400):
        intensity = int(255 * i / 400)
        image[:, i] = intensity
    
    # إضافة أشكال بألوان مختلفة
    cv2.rectangle(image, (50, 50), (150, 150), 100, -1)
    cv2.circle(image, (300, 100), 50, 200, -1)
    cv2.line(image, (200, 50), (250, 200), 150, 3)
    
    # إضافة نص
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Grayscale', (100, 250), font, 1, 255, 2, cv2.LINE_AA)
    
    return image

def apply_gamma_correction(image, gamma):
    """تطبيق Gamma Correction على الصورة"""
    # بناء lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    # تطبيق gamma correction
    return cv2.LUT(image, table)