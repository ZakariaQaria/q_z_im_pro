import streamlit as st
import cv2
import numpy as np
from helpers import load_image, add_user_progress
from achievements import add_achievement  # ⬅️ أضف هذا السطر

def show_module4():
    """عرض المحاضرة الرابعة: الفلاتر والالتفاف"""
    
    st.header("🔍 المحاضرة 4: الفلاتر والالتفاف (Filtering & Convolution)")
    
    # معلومات التقدم
    if st.session_state.progress.get("module4", False):
        st.success("✅ لقد أكملت هذه المحاضرة بالفعل")
    else:
        if add_user_progress("module4"):
            st.balloons()
            st.success("🎉 مبروك! لقد أكملت المحاضرة الرابعة وحصلت على 20 نقطة")
            add_achievement("سيد الفلاتر", "إكمال وحدة الفلاتر والالتفاف")
    
    # النظرية
    with st.expander("📖 الشرح النظري", expanded=True):
        st.markdown("""
        ## الفلاتر والالتفاف (Filtering & Convolution)

        ### مفهوم النواة (Kernel)
        النواة هي مصفوفة صغيرة تستخدم لتنفيذ عمليات مختلفة على الصورة عن طريق الالتفاف (Convolution).

        ### عملية الالتفاف (Convolution)
        - **المبدأ**: انزلاق النواة على الصورة وإجراء عملية حسابية في كل موضع
        - **الصيغة**: لكل بكسل، يتم حساب weighted sum من البكسلات المجاورة
        - **المعادلة**: 
          ```
          output[i, j] = sum_{k,l} input[i+k, j+l] * kernel[k, l]
          ```

        ### أنواع الفلاتر الشائعة:

        #### 1. فلتر التمويه (Blur)
        - **الغرض**: تقليل الضوضاء وتنعيم الصورة
        - **أنواعه**:
          * **Average Blur**: متوسط القيم المجاورة
          * **Gaussian Blur**: weighted average مع تركيز على المركز
          * **Median Blur**: الوسيط الحسابي - جيد للضوضاء الملح والفلفل

        #### 2. فلتر الحدة (Sharpening)
        - **الغرض**: زيادة وضوح الحواف والتفاصيل
        - **المبدأ**: إضافة جزء من high-frequency content للصورة

        #### 3. فلتر الكشف عن الحواف (Edge Detection)
        - **الغرض**: إبراز المناطق التي يتغير فيها اللون أو الشدة suddenly
        - **أمثلة**: Sobel, Prewitt, Laplacian

        #### 4. فلتر التغميق (Emboss)
        - **الغرض**: إعطاء تأثير ثلاثي الأبعاد مثل النقش

        ### خصائص النواة المهمة:
        - **الحجم**: عادة 3x3, 5x5, 7x7 (كلما كبر الحجم زاد تأثير التمويه)
        - **القيم**: تحدد نوع التأثير المطلوب
        - **التطبيع**: يجب أن مجموع قيم النواة = 1 للحفاظ على سطوع الصورة
        """)
    
    st.markdown("---")
    
    # التطبيق العملي
    st.subheader("🔧 التجربة العملية: تطبيق الفلاتر المختلفة")
    
    # تحميل الصورة
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📤 تحميل الصورة")
        uploaded_file = st.file_uploader("اختر صورة", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = load_image(uploaded_file)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            st.image(image, caption="الصورة الأصلية", use_container_width=True)
        else:
            # إنشاء صورة افتراضية
            image = create_detailed_sample_image()
            st.image(image, caption="الصورة الافتراضية", use_container_width=True)
    
    with col2:
        if image is not None:
            st.markdown("#### ⚙️ اختر نوع الفلتر")
            
            filter_type = st.selectbox(
                "نوع الفلتر:",
                ["Gaussian Blur", "Median Blur", "Sharpening", "Sobel Edge", "Laplacian", "Emboss", "مخصص"]
            )
            
            if filter_type in ["Gaussian Blur", "Median Blur"]:
                kernel_size = st.slider("حجم النواة", 3, 15, 5, 2)
                if kernel_size % 2 == 0:  # يجب أن يكون فردياً
                    kernel_size += 1
            
            if filter_type == "Gaussian Blur":
                if st.button("🔄 تطبيق Gaussian Blur"):
                    result = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
                    st.image(result, caption=f"Gaussian Blur (حجم {kernel_size}x{kernel_size})", use_container_width=True)
                    show_kernel_info("Gaussian", kernel_size)
            
            elif filter_type == "Median Blur":
                if st.button("🔄 تطبيق Median Blur"):
                    result = cv2.medianBlur(image, kernel_size)
                    st.image(result, caption=f"Median Blur (حجم {kernel_size}x{kernel_size})", use_container_width=True)
                    show_kernel_info("Median", kernel_size)
            
            elif filter_type == "Sharpening":
                strength = st.slider("قوة الحدة", 0.1, 3.0, 1.0, 0.1)
                if st.button("🔄 تطبيق Sharpening"):
                    result = apply_sharpening(image, strength)
                    st.image(result, caption=f"Sharpening (قوة {strength})", use_container_width=True)
                    show_kernel_info("Sharpening", 3)
            
            elif filter_type == "Sobel Edge":
                direction = st.selectbox("الاتجاه", ["X", "Y", "كلاهما"])
                if st.button("🔄 تطبيق Sobel"):
                    result = apply_sobel(image, direction)
                    st.image(result, caption=f"Sobel {direction}", use_container_width=True)
                    show_kernel_info("Sobel", 3)
            
            elif filter_type == "Laplacian":
                if st.button("🔄 تطبيق Laplacian"):
                    result = cv2.Laplacian(image, cv2.CV_64F)
                    result = np.uint8(np.absolute(result))
                    st.image(result, caption="Laplacian Edge Detection", use_container_width=True)
                    show_kernel_info("Laplacian", 3)
            
            elif filter_type == "Emboss":
                if st.button("🔄 تطبيق Emboss"):
                    result = apply_emboss(image)
                    st.image(result, caption="Emboss Effect", use_container_width=True)
                    show_kernel_info("Emboss", 3)
            
            elif filter_type == "مخصص":
                st.markdown("##### 🛠️ إنشاء نواة مخصصة")
                st.write("أدخل قيم النواة 3x3:")
                
                cols = st.columns(3)
                kernel = []
                
                for i in range(3):
                    row = []
                    for j in range(3):
                        with cols[j]:
                            value = st.number_input(f"[{i},{j}]", value=0.0 if i==j else 0.0, 
                                                  key=f"k_{i}_{j}", format="%.1f")
                            row.append(value)
                    kernel.append(row)
                
                kernel = np.array(kernel)
                
                if st.button("🔄 تطبيق النواة المخصصة"):
                    # تطبيع النواة إذا كان المجموع ليس صفراً
                    if np.sum(kernel) != 0:
                        kernel = kernel / np.sum(kernel)
                    
                    result = cv2.filter2D(image, -1, kernel)
                    st.image(result, caption="بعد تطبيق النواة المخصصة", use_container_width=True)
                    
                    st.markdown("**النواة المستخدمة:**")
                    st.write(kernel)
    
    st.markdown("---")
    
    # قسم إضافي: مقارنة بين الفلاتر
    st.subheader("📊 مقارنة بين تأثير الفلاتر المختلفة")
    
    if image is not None:
        st.markdown("##### تأثير أحجام نواة مختلفة على Gaussian Blur:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            blur3 = cv2.GaussianBlur(image, (3, 3), 0)
            st.image(blur3, caption="حجم 3x3", use_container_width=True)
        
        with col2:
            blur7 = cv2.GaussianBlur(image, (7, 7), 0)
            st.image(blur7, caption="حجم 7x7", use_container_width=True)
        
        with col3:
            blur11 = cv2.GaussianBlur(image, (11, 11), 0)
            st.image(blur11, caption="حجم 11x11", use_container_width=True)
    
    # اختبار قصير
    st.markdown("---")
    with st.expander("🧪 اختبار قصير", expanded=False):
        st.subheader("اختبار فهم الفلاتر والالتفاف")
        
        q1 = st.radio(
            "1. ما هو الغرض الرئيسي من فلتر Gaussian Blur?",
            ["زيادة حدة الصورة", "تقليل الضوضاء وتنعيم الصورة", "كشف الحواف", "تغميق الصورة"]
        )
        
        q2 = st.radio(
            "2. لماذا يجب أن يكون حجم النواة فردياً في معظم الفلاتر?",
            ["لتحسين السرعة", "لأن الصور مربعة", "لوجود مركز واضح للنواة", "لا يوجد سبب"]
        )
        
        q3 = st.radio(
            "3. أي فلتر يستخدم للكشف عن الحواف?",
            ["Gaussian Blur", "Median Blur", "Sobel", "Emboss"]
        )
        
        if st.button("✅ التحقق من الإجابات"):
            score = 0
            if q1 == "تقليل الضوضاء وتنعيم الصورة":
                score += 1
                st.success("السؤال 1: صحيح! Gaussian Blur ينعم الصورة ويقلل الضوضاء")
            else:
                st.error("السؤال 1: خطأ! الغرض الرئيسي هو تقليل الضوضاء وتنعيم الصورة")
            
            if q2 == "لوجود مركز واضح للنواة":
                score += 1
                st.success("السؤال 2: صحيح! الحجم الفردي يضمن وجود مركز واضح")
            else:
                st.error("السؤال 2: خطأ! الحجم الفردي يضمن وجود مركز واضح للنواة")
            
            if q3 == "Sobel":
                score += 1
                st.success("السؤال 3: صحيح! Sobel هو فلتر لكشف الحواف")
            else:
                st.error("السؤال 3: خطأ! Sobel هو المستخدم للكشف عن الحواف")
            
            st.info(f"**النتيجة: {score}/3**")
            
            if score == 3:
                st.balloons()
                st.session_state.quiz_score = st.session_state.get('quiz_score', 0) + 3

def create_detailed_sample_image():
    """إنشاء صورة تفصيلية للفلاتر"""
    image = np.zeros((300, 400), dtype=np.uint8)
    
    # إضافة تدرج
    for i in range(400):
        intensity = int(255 * i / 400)
        image[:, i] = intensity
    
    # إضافة حواف وأشكال مختلفة
    cv2.rectangle(image, (50, 50), (150, 150), 100, -1)
    cv2.circle(image, (300, 100), 50, 200, -1)
    cv2.line(image, (200, 50), (250, 200), 150, 3)
    
    # إضافة نص
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Filters', (100, 250), font, 1, 255, 2, cv2.LINE_AA)
    
    # إضافة ضوضاء
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    
    return image

def apply_sharpening(image, strength=1.0):
    """تطبيق فلتر Sharpening"""
    # نواة الحدة
    kernel = np.array([[-1, -1, -1],
                       [-1, 9*strength, -1],
                       [-1, -1, -1]])
    
    return cv2.filter2D(image, -1, kernel)

def apply_sobel(image, direction="X"):
    """تطبيق فلتر Sobel"""
    if direction == "X":
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    elif direction == "Y":
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    else:  # كلاهما
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    
    return np.uint8(np.absolute(sobel))

def apply_emboss(image):
    """تطبيق فلتر Emboss"""
    kernel = np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]])
    
    emboss = cv2.filter2D(image, -1, kernel)
    # إضافة offset لجعل القيم موجبة
    emboss = emboss + 128
    return np.clip(emboss, 0, 255).astype(np.uint8)

def show_kernel_info(kernel_type, size):
    """عرض معلومات عن النواة المستخدمة"""
    st.markdown("**معلومات النواة:**")
    
    if kernel_type == "Gaussian":
        st.write(f"نوع النواة: Gaussian ({size}x{size})")
        st.write("تأثير: تمويه مع وزن أكبر للبكسلات القريبة من المركز")
    
    elif kernel_type == "Median":
        st.write(f"نوع النواة: Median ({size}x{size})")
        st.write("تأثير: يأخذ الوسيط الحسابي - ممتاز للضوضاء الملح والفلفل")
    
    elif kernel_type == "Sharpening":
        st.write("نوع النواة: Sharpening (3x3)")
        st.write("تأثير: يزيد حدة الحواف والتفاصيل")
    
    elif kernel_type == "Sobel":
        st.write("نوع النواة: Sobel (3x3)")
        st.write("تأثير: يكشف الحواف في اتجاه محدد")
    
    elif kernel_type == "Laplacian":
        st.write("نوع النواة: Laplacian (3x3)")
        st.write("تأثير: يكشف الحواف في جميع الاتجاهات")
    
    elif kernel_type == "Emboss":
        st.write("نوع النواة: Emboss (3x3)")
        st.write("تأثير: يعطي تأثير ثلاثي الأبعاد مثل النقش")