import streamlit as st
import cv2
import numpy as np
from helpers import load_image, add_user_progress

def show_module7():
    """عرض المحاضرة السابعة: العمليات المورفولوجية"""
    
    st.header("🔲 المحاضرة 7: العمليات المورفولوجية (Morphological Operations)")
    
    # معلومات التقدم
    if st.session_state.progress.get("module7", False):
        st.success("✅ لقد أكملت هذه المحاضرة بالفعل")
    else:
        if add_user_progress("module7"):
            st.balloons()
            st.success("🎉 مبروك! لقد أكملت المحاضرة السابعة وحصلت على 20 نقطة")
    
    # النظرية
    with st.expander("📖 الشرح النظري", expanded=True):
        st.markdown("""
        ## العمليات المورفولوجية (Morphological Operations)

        ### ما هي العمليات المورفولوجية؟
        هي عمليات تعتمد على الشكل (morphology) وتطبق على الصور الثنائية (أبيض وأسود) ل:
        - تحسين شكل الأشياء
        - إزالة الضوضاء
        - استخراج مكونات الصورة
        - تحسين جودة الصورة

        ### العنصر البنائي (Structuring Element)
        - **التعريف**: نواة (kernel) صغيرة تحدد شكل العملية المورفولوجية
        - **الأشكال الشائعة**:
          * مستطيل (Rectangle)
          * بيضاوي (Ellipse)
          * صليب (Cross)
        - **الحجم**: يحدد قوة التأثير

        ### العمليات الأساسية:

        #### 1. التآكل (Erosion)
        - **المبدأ**: تقليص حدود الأشياء البيضاء
        - **التأثير**: إزالة النقاط الصغيرة والعزلة
        - **الصيغة**: `A ⊖ B = {z | (B)_z ⊆ A}`
        - **الاستخدام**: فصل objects متصلة، إزالة noise صغير

        #### 2. التمدد (Dilation)
        - **المبدأ**: توسيع حدود الأشياء البيضاء
        - **التأثير**: سد الفجوات الصغيرة، ربط الأجزاء المنفصلة
        - **الصيغة**: `A ⊕ B = {z | (B̂)_z ∩ A ≠ ∅}`
        - **الاستخدام**: ملء الثقوب، ربط الخطوط المتقطعة

        #### 3. الفتح (Opening)
        - **المبدأ**: تآكل ثم تمدد
        - **التأثير**: إزالة الأجسام الصغيرة مع الحفاظ على شكل الكبيرة
        - **الصيغة**: `A ∘ B = (A ⊖ B) ⊕ B`
        - **الاستخدام**: إزالة الضوضاء البيضاء على خلفية سوداء

        #### 4. الإغلاق (Closing)
        - **المبدأ**: تمدد ثم تآكل
        - **التأثير**: سد الثقوب الصغيرة مع الحفاظ على شكل الأشياء
        - **الصيغة**: `A • B = (A ⊕ B) ⊖ B`
        - **الاستخدام**: إزالة الثقوب السوداء على خلفية بيضاء

        ### عمليات متقدمة:

        #### 1. Gradient المورفولوجي
        - **المبدأ**: الفرق بين التمدد والتآكل
        - **النتيجة**: outline of objects

        #### 2. Top Hat
        - **المبدأ**: الفرق بين الصورة والفتح
        - **الاستخدام**: إبراز العناصر الساطعة على خلفية مظلمة

        #### 3. Black Hat
        - **المبدأ**: الفرق بين الإغلاق والصورة
        - **الاستخدام**: إبراز العناصر المظلمة على خلفية ساطعة

        ### تطبيقات عملية:
        - معالجة الصور الطبية (الأشعة)
        - التعرف على الشخصيات (OCR)
        - تحليل الصور الجوية والفضائية
        - فحص الجودة في التصنيع
        """)
    
    st.markdown("---")
    
    # التطبيق العملي
    st.subheader("🔧 التجربة العملية: العمليات المورفولوجية")
    
    # تحميل الصورة
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📤 تحميل الصورة")
        uploaded_file = st.file_uploader("اختر صورة", type=['jpg', 'jpeg', 'png'], key="morph_upload")
        
        if uploaded_file is not None:
            original_image = load_image(uploaded_file)
            if len(original_image.shape) == 3:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            st.image(original_image, caption="الصورة الأصلية", use_container_width=True)
        else:
            # إنشاء صورة افتراضية
            original_image = create_morphological_sample()
            st.image(original_image, caption="الصورة الافتراضية", use_container_width=True)
    
    with col2:
        if original_image is not None:
            st.markdown("#### ⚙️ تحويل إلى صورة ثنائية")
            
            # تحويل إلى صورة ثنائية
            threshold = st.slider("قيمة العتبة", 0, 255, 127)
            _, binary_image = cv2.threshold(original_image, threshold, 255, cv2.THRESH_BINARY)
            st.image(binary_image, caption="الصورة الثنائية", use_container_width=True)
            
            st.markdown("#### 🛠️ اختر العملية المورفولوجية")
            
            operation = st.selectbox(
                "العملية:",
                ["التآكل (Erosion)", "التمدد (Dilation)", "الفتح (Opening)", 
                 "الإغلاق (Closing)", "Gradient", "Top Hat", "Black Hat", "جميع العمليات"]
            )
            
            # إعداد العنصر البنائي
            kernel_shape = st.selectbox("شكل النواة:", ["مستطيل", "بيضاوي", "صليب"])
            kernel_size = st.slider("حجم النواة", 1, 15, 3, 2)
            
            if st.button("🔄 تطبيق العملية"):
                with st.spinner("جاري المعالجة..."):
                    # إنشاء النواة
                    if kernel_shape == "مستطيل":
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
                    elif kernel_shape == "بيضاوي":
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    else:  # صليب
                        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
                    
                    # تطبيق العملية المحددة
                    if operation == "التآكل (Erosion)":
                        result = cv2.erode(binary_image, kernel, iterations=1)
                    elif operation == "التمدد (Dilation)":
                        result = cv2.dilate(binary_image, kernel, iterations=1)
                    elif operation == "الفتح (Opening)":
                        result = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
                    elif operation == "الإغلاق (Closing)":
                        result = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
                    elif operation == "Gradient":
                        result = cv2.morphologyEx(binary_image, cv2.MORPH_GRADIENT, kernel)
                    elif operation == "Top Hat":
                        result = cv2.morphologyEx(binary_image, cv2.MORPH_TOPHAT, kernel)
                    elif operation == "Black Hat":
                        result = cv2.morphologyEx(binary_image, cv2.MORPH_BLACKHAT, kernel)
                    elif operation == "جميع العمليات":
                        result = None
                    
                    if operation != "جميع العمليات":
                        st.image(result, caption=operation, use_container_width=True)
                        
                        # إحصائيات
                        white_pixels = np.sum(result == 255)
                        black_pixels = np.sum(result == 0)
                        st.metric("البكسلات البيضاء", white_pixels)
                        st.metric("البكسلات السوداء", black_pixels)
                    else:
                        # عرض جميع العمليات
                        show_all_morphological_ops(binary_image, kernel)
    
    st.markdown("---")
    
    # قسم إضافي: تطبيقات عملية
    if original_image is not None:
        st.subheader("💼 تطبيقات عملية للعمليات المورفولوجية")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🔍 استخراج الحدود")
            
            # استخراج الحدود باستخدام Gradient
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            gradient = cv2.morphologyEx(binary_image, cv2.MORPH_GRADIENT, kernel)
            st.image(gradient, caption="الحدود المستخرجة", use_container_width=True)
        
        with col2:
            st.markdown("##### 🎯 عزل الأشياء")
            
            # فتح لإزالة الضوضاء الصغيرة
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
            st.image(cleaned, caption="بعد إزالة الضوضاء", use_container_width=True)
    
    # اختبار قصير
    st.markdown("---")
    with st.expander("🧪 اختبار قصير", expanded=False):
        st.subheader("اختبار فهم العمليات المورفولوجية")
        
        q1 = st.radio(
            "1. ما هو الترتيب الصحيح للفتح (Opening)?",
            ["تمدد ثم تآكل", "تآكل ثم تمدد", "تآكل فقط", "تمدد فقط"],
            key="morph_q1"
        )
        
        q2 = st.radio(
            "2. أي عملية تستخدم لسد الثقوب الصغيرة?",
            ["التآكل", "التمدد", "الفتح", "الإغلاق"],
            key="morph_q2"
        )
        
        q3 = st.radio(
            "3. ما هو تأثير عملية Top Hat?",
            ["إبراز العناصر الساطعة", "إبراز العناصر المظلمة", "توسيع الحدود", "تقليص الحدود"],
            key="morph_q3"
        )
        
        if st.button("✅ التحقق من الإجابات", key="morph_check"):
            score = 0
            if q1 == "تآكل ثم تمدد":
                score += 1
                st.success("السؤال 1: صحيح! الفتح = تآكل ثم تمدد")
            else:
                st.error("السؤال 1: خطأ! الفتح هو تآكل ثم تمدد")
            
            if q2 == "الإغلاق":
                score += 1
                st.success("السؤال 2: صحيح! الإغلاق يستخدم لسد الثقوب")
            else:
                st.error("السؤال 2: خطأ! الإغلاق هو الذي يسد الثقوب")
            
            if q3 == "إبراز العناصر الساطعة":
                score += 1
                st.success("السؤال 3: صحيح! Top Hat يبرز العناصر الساطعة")
            else:
                st.error("السؤال 3: خطأ! Top Hat يبرز العناصر الساطعة")
            
            st.info(f"**النتيجة: {score}/3**")
            
            if score == 3:
                st.balloons()
                st.session_state.quiz_score = st.session_state.get('quiz_score', 0) + 3

def create_morphological_sample():
    """إنشاء صورة مناسبة للعمليات المورفولوجية"""
    image = np.zeros((300, 400), dtype=np.uint8)
    
    # إضافة أشكال مختلفة مع ثقوب وضوضاء
    cv2.rectangle(image, (50, 50), (150, 150), 255, -1)
    cv2.circle(image, (300, 100), 50, 255, -1)
    
    # إضافة ثقوب داخل الأشكال
    cv2.circle(image, (300, 100), 20, 0, -1)  # ثقب في الدائرة
    cv2.rectangle(image, (80, 80), (120, 120), 0, -1)  # ثقب في المربع
    
    # إضافة ضوضاء (نقاط بيضاء وسوداء صغيرة)
    for _ in range(100):
        x, y = np.random.randint(0, 400), np.random.randint(0, 300)
        if np.random.rand() > 0.5:
            image[y, x] = 255  # نقطة بيضاء
        else:
            image[y, x] = 0    # نقطة سوداء
    
    # إضافة خطوط متقطعة
    for i in range(0, 400, 20):
        cv2.line(image, (i, 200), (i + 10, 200), 255, 2)
    
    return image

def show_all_morphological_ops(binary_image, kernel):
    """عرض جميع العمليات المورفولوجية"""
    st.markdown("##### 📊 مقارنة بين جميع العمليات المورفولوجية")
    
    operations = {
        "الأصل": binary_image,
        "التآكل": cv2.erode(binary_image, kernel),
        "التمدد": cv2.dilate(binary_image, kernel),
        "الفتح": cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel),
        "الإغلاق": cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel),
        "Gradient": cv2.morphologyEx(binary_image, cv2.MORPH_GRADIENT, kernel),
        "Top Hat": cv2.morphologyEx(binary_image, cv2.MORPH_TOPHAT, kernel),
        "Black Hat": cv2.morphologyEx(binary_image, cv2.MORPH_BLACKHAT, kernel)
    }
    
    cols = st.columns(4)
    col_idx = 0
    
    for name, result in operations.items():
        with cols[col_idx]:
            st.image(result, caption=name, use_container_width=True)
            white_pixels = np.sum(result == 255)
            st.caption(f"بيضاء: {white_pixels}")
        
        col_idx = (col_idx + 1) % 4
        if col_idx == 0 and name != "Black Hat":
            st.markdown("---")