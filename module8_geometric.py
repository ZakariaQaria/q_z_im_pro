import streamlit as st
import cv2
import numpy as np
from helpers import load_image, add_user_progress

def show_module8():
    """عرض المحاضرة الثامنة: التحويلات الهندسية"""
    
    st.header("📐 المحاضرة 8: التحويلات الهندسية (Geometric Transforms)")
    
    # معلومات التقدم
    if st.session_state.progress.get("module8", False):
        st.success("✅ لقد أكملت هذه المحاضرة بالفعل")
    else:
        if add_user_progress("module8"):
            st.balloons()
            st.success("🎉 مبروك! لقد أكملت المحاضرة الثامنة وحصلت على 20 نقطة")
    
    # النظرية
    with st.expander("📖 الشرح النظري", expanded=True):
        st.markdown("""
        ## التحويلات الهندسية (Geometric Transformations)

        ### ما هي التحويلات الهندسية؟
        هي عمليات تغير الهندسة المكانية للصورة (المواضع النسبية للبكسلات) مع الحفاظ على محتوى الصورة.

        ### أنواع التحويلات الأساسية:

        #### 1. الإزاحة (Translation)
        - **المبدأ**: تحريك الصورة في اتجاه معين
        - **المعادلة**: 
          ```
          x' = x + t_x
          y' = y + t_y
          ```
        - **الاستخدام**: محاذاة الصور، تصحيح المواضع

        #### 2. الدوران (Rotation)
        - **المبدأ**: تدوير الصورة حول نقطة مركزية
        - **المعادلة**:
          ```
          x' = x⋅cosθ - y⋅sinθ
          y' = x⋅sinθ + y⋅cosθ
          ```
        - **الاستخدام**: تصحيح الاتجاه، زيادة البيانات

        #### 3. القياس (Scaling)
        - **المبدأ**: تكبير أو تصغير الصورة
        - **المعادلة**:
          ```
          x' = s_x ⋅ x
          y' = s_y ⋅ y
          ```
        - **الاستخدام**: تغيير الحجم، تحسين الدقة

        #### 4. القص (Shearing)
        - **المبدأ**: إزاحة غير منتظمة تحول المستطيل إلى متوازي أضلاع
        - **الاستخدام**: تصحيح التشوهات، تأثيرات بصرية

        ### التحويلات المتقدمة:

        #### 1. التحويل الأفيني (Affine Transform)
        - **المبدأ**: يحافظ على الخطوط المتوازية (translation + rotation + scaling + shearing)
        - **الدرجات الحرية**: 6
        - **المعادلة**: تحتاج 3 points for mapping

        #### 2. التحويل الإسقاطي (Projective Transform/Homography)
        - **المبدأ**: يحافظ على الخطوط المستقيمة (لا يحافظ على التوازي)
        - **الدرجات الحرية**: 8
        - **المعادلة**: تحتاج 4 points for mapping
        - **الاستخدام**: تصحيح perspective، panoramas

        ### طرق Interpolation الهامة:

        #### 1. Nearest Neighbor
        - **المبدأ**: أخذ قيمة أقرب بكسل
        - **المميزات**: سريع
        - **العيوب**: pixelated results

        #### 2. Bilinear Interpolation
        - **المبدأ**: متوسط مر weighted لمتوسط 4 بكسلات مجاورة
        - **المميزات**: smoother من Nearest Neighbor
        - **العيوب**: أكثر بطئاً

        #### 3. Bicubic Interpolation
        - **المبدأ**: استخدام 16 بكسل مجاور
        - **المميزات**: أعلى جودة
        - **العيوب**: الأبطأ

        ### تطبيقات عملية:
        - تصحيح تشوهات الكاميرا
        - تركيب الصور (Image registration)
        - الواقع المعزز
        - معالجة الصور الطبية
        """)
    
    st.markdown("---")
    
    # التطبيق العملي
    st.subheader("🔧 التجربة العملية: التحويلات الهندسية")
    
    # تحميل الصورة
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📤 تحميل الصورة")
        uploaded_file = st.file_uploader("اختر صورة", type=['jpg', 'jpeg', 'png'], key="geo_upload")
        
        if uploaded_file is not None:
            image = load_image(uploaded_file)
            st.image(image, caption="الصورة الأصلية", use_container_width=True)
        else:
            # إنشاء صورة افتراضية
            image = create_geometric_sample()
            st.image(image, caption="الصورة الافتراضية", use_container_width=True)
    
    with col2:
        if image is not None:
            st.markdown("#### ⚙️ اختر نوع التحويل")
            
            transform_type = st.selectbox(
                "نوع التحويل:",
                ["الإزاحة", "الدوران", "القياس", "الانعكاس", "القص", "Affine", "Homography", "جميع التحويلات"]
            )
            
            # إعداد المعاملات حسب نوع التحويل
            if transform_type == "الإزاحة":
                tx = st.slider("الإزاحة الأفقية", -100, 100, 0)
                ty = st.slider("الإزاحة الرأسية", -100, 100, 0)
            
            elif transform_type == "الدوران":
                angle = st.slider("زاوية الدوران", -180, 180, 0)
                center_x = st.slider("مركز الدوران X", 0, image.shape[1], image.shape[1]//2)
                center_y = st.slider("مركز الدوران Y", 0, image.shape[0], image.shape[0]//2)
            
            elif transform_type == "القياس":
                scale_x = st.slider("مقياس العرض", 0.1, 3.0, 1.0, 0.1)
                scale_y = st.slider("مقياس الارتفاع", 0.1, 3.0, 1.0, 0.1)
                keep_aspect = st.checkbox("الحفاظ على نسبة الأبعاد", value=True)
            
            elif transform_type == "الانعكاس":
                flip_code = st.radio("اتجاه الانعكاس:", ["افقي", "رأسي", "كلاهما"])
            
            elif transform_type == "القص":
                shear_x = st.slider("قص أفقي", -1.0, 1.0, 0.0, 0.1)
                shear_y = st.slider("قص رأسي", -1.0, 1.0, 0.0, 0.1)
            
            elif transform_type in ["Affine", "Homography"]:
                st.info("حدد نقاط التحويل باستخدام المؤشر")
                points = []
            
            # اختيار طريقة Interpolation
            interp_method = st.selectbox("طريقة الاستيفاء:", ["Nearest Neighbor", "Bilinear", "Bicubic"])
            
            if st.button("🔄 تطبيق التحويل"):
                with st.spinner("جاري المعالجة..."):
                    result = apply_geometric_transform(image, transform_type, locals())
                    
                    if transform_type != "جميع التحويلات":
                        st.image(result, caption=f"بعد {transform_type}", use_container_width=True)
                    else:
                        # عرض جميع التحويلات
                        show_all_geometric_transforms(image)
    
    st.markdown("---")
    
    # قسم إضافي: تأثير طرق الاستيفاء
    if image is not None:
        st.subheader("🔍 مقارنة طرق الاستيفاء (Interpolation)")
        
        # تطبيق نفس التحويل بطرق استيفاء مختلفة
        methods = {
            "Nearest Neighbor": cv2.INTER_NEAREST,
            "Bilinear": cv2.INTER_LINEAR,
            "Bicubic": cv2.INTER_CUBIC
        }
        
        cols = st.columns(3)
        for (name, method), col in zip(methods.items(), cols):
            with col:
                # تطبيق تحويل قياسي
                rows, cols = image.shape[:2]
                M = np.float32([[1, 0.2, 0], [0.2, 1, 0]])
                transformed = cv2.warpAffine(image, M, (cols, cols), flags=method)
                st.image(transformed, caption=name, use_container_width=True)
                
                # حساب جودة الاستيفاء
                if len(image.shape) == 2:
                    sharpness = calculate_sharpness(transformed)
                    st.metric("حدة الصورة", f"{sharpness:.2f}")
    
    # اختبار قصير
    st.markdown("---")
    with st.expander("🧪 اختبار قصير", expanded=False):
        st.subheader("اختبار فهم التحويلات الهندسية")
        
        q1 = st.radio(
            "1. كم نقطة يحتاجها التحويل الأفيني?",
            ["نقطتين", "ثلاث نقاط", "أربع نقاط", "خمس نقاط"],
            key="geo_q1"
        )
        
        q2 = st.radio(
            "2. أي طريقة استيفاء تعطي أفضل جودة?",
            ["Nearest Neighbor", "Bilinear", "Bicubic", "كلهم نفس الجودة"],
            key="geo_q2"
        )
        
        q3 = st.radio(
            "3. ما هو التحويل الذي يحافظ على الخطوط المتوازية?",
            ["الإزاحة فقط", "الدوران فقط", "التحويل الأفيني", "التحويل الإسقاطي"],
            key="geo_q3"
        )
        
        if st.button("✅ التحقق من الإجابات", key="geo_check"):
            score = 0
            if q1 == "ثلاث نقاط":
                score += 1
                st.success("السؤال 1: صحيح! الأفيني يحتاج 3 نقاط")
            else:
                st.error("السؤال 1: خطأ! الأفيني يحتاج 3 نقاط")
            
            if q2 == "Bicubic":
                score += 1
                st.success("السؤال 2: صحيح! Bicubic يعطي أفضل جودة")
            else:
                st.error("السؤال 2: خطأ! Bicubic هو الأعلى جودة")
            
            if q3 == "التحويل الأفيني":
                score += 1
                st.success("السؤال 3: صحيح! الأفيني يحافظ على التوازي")
            else:
                st.error("السؤال 3: خطأ! الأفيني يحافظ على الخطوط المتوازية")
            
            st.info(f"**النتيجة: {score}/3**")
            
            if score == 3:
                st.balloons()
                st.session_state.quiz_score = st.session_state.get('quiz_score', 0) + 3

def create_geometric_sample():
    """إنشاء صورة مناسبة للتحويلات الهندسية"""
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # إضافة شبكة من الخطوط
    for i in range(0, 400, 20):
        cv2.line(image, (i, 0), (i, 300), (255, 255, 255), 1)
    for i in range(0, 300, 20):
        cv2.line(image, (0, i), (400, i), (255, 255, 255), 1)
    
    # إضافة أشكال ملونة
    cv2.rectangle(image, (50, 50), (150, 150), (0, 0, 255), -1)  # أحمر
    cv2.circle(image, (300, 100), 50, (0, 255, 0), -1)  # أخضر
    cv2.rectangle(image, (180, 200), (280, 280), (255, 0, 0), -1)  # أزرق
    
    # إضافة نص
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Geometry', (150, 250), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return image

def apply_geometric_transform(image, transform_type, params):
    """تطبيق التحويل الهندسي المحدد"""
    h, w = image.shape[:2]
    
    if transform_type == "الإزاحة":
        M = np.float32([[1, 0, params['tx']], [0, 1, params['ty']]])
        return cv2.warpAffine(image, M, (w, h))
    
    elif transform_type == "الدوران":
        center = (params['center_x'], params['center_y'])
        M = cv2.getRotationMatrix2D(center, params['angle'], 1)
        return cv2.warpAffine(image, M, (w, h))
    
    elif transform_type == "القياس":
        if params['keep_aspect']:
            scale = min(params['scale_x'], params['scale_y'])
            new_w, new_h = int(w * scale), int(h * scale)
        else:
            new_w, new_h = int(w * params['scale_x']), int(h * params['scale_y'])
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    elif transform_type == "الانعكاس":
        code = 1 if params['flip_code'] == "افقي" else 0
        if params['flip_code'] == "كلاهما":
            code = -1
        return cv2.flip(image, code)
    
    elif transform_type == "القص":
        M = np.float32([[1, params['shear_x'], 0], [params['shear_y'], 1, 0]])
        return cv2.warpAffine(image, M, (w, h))
    
    elif transform_type == "Affine":
        # نقاط افتراضية للتحويل الأفيني
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        M = cv2.getAffineTransform(pts1, pts2)
        return cv2.warpAffine(image, M, (w, h))
    
    elif transform_type == "Homography":
        # نقاط افتراضية للتحويل الإسقاطي
        pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
        pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(image, M, (300, 300))
    
    return image

def show_all_geometric_transforms(image):
    """عرض جميع التحويلات الهندسية"""
    st.markdown("##### 📊 مقارنة بين التحويلات الهندسية الأساسية")
    
    transforms = {
        "الأصل": image,
        "الإزاحة": apply_geometric_transform(image, "الإزاحة", {'tx': 30, 'ty': -20}),
        "الدوران": apply_geometric_transform(image, "الدوران", {'angle': 45, 'center_x': 200, 'center_y': 150}),
        "القياس": cv2.resize(image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC),
        "انعكاس أفقي": cv2.flip(image, 1),
        "انعكاس رأسي": cv2.flip(image, 0)
    }
    
    cols = st.columns(3)
    col_idx = 0
    
    for name, result in transforms.items():
        with cols[col_idx]:
            st.image(result, caption=name, use_container_width=True)
        
        col_idx = (col_idx + 1) % 3
        if col_idx == 0 and name != "انعكاس رأسي":
            st.markdown("---")

def calculate_sharpness(image):
    """حساب حدة الصورة باستخدام Laplacian variance"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(image, cv2.CV_64F).var()