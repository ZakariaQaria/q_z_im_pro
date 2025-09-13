import streamlit as st
import cv2
import numpy as np
from helpers import load_image, add_user_progress
from achievements import add_achievement  # ⬅️ هذا السطر الناقص!

def show_module6():
    """عرض المحاضرة السادسة: كشف الحواف"""
    
    st.header("📐 المحاضرة 6: كشف الحواف (Edge Detection)")
    
    # معلومات التقدم
    if st.session_state.progress.get("module6", False):
        st.success("✅ لقد أكملت هذه المحاضرة بالفعل")
    else:
        if add_user_progress("module6"):
            st.balloons()
            st.success("🎉 مبروك! لقد أكملت المحاضرة السادسة وحصلت على 20 نقطة")
            add_achievement("كاشف الحواف", "إكمال وحدة كشف الحواف")
    
    # النظرية
    with st.expander("📖 الشرح النظري", expanded=True):
        st.markdown("""
        ## كشف الحواف في الصور (Edge Detection)

        ### ما هي الحواف؟
        الحواف هي مناطق في الصورة حيث تتغير شدة اللون بشكل مفاجئ (فجائي). تمثل هذه المناطق عادة:
        - حدود between objects
        - تغيرات في الإضاءة
        - تغيرات في الملمس
        - حدود هندسية

        ### لماذا نكشف الحواف؟
        - **تحديد الأشياء**: تحديد حدود و contornos الأشياء
        - **تخفيض البيانات**: الحواف تمثل معلومات مهمة بحجم أصغر
        - **الرؤية الحاسوبية**: خطوة أساسية في many computer vision algorithms
        - **التعرف على الأنماط**: helpful في pattern recognition

        ### طرق كشف الحواف:

        #### 1. مشتقات الدرجة الأولى (First-order derivatives)
        - **المبدأ**: قياس gradient (معدل التغير) في الشدة
        - **أمثلة**: Sobel, Prewitt, Roberts
        - **الكشف**: عن الحواف based on maximum gradient

        #### 2. مشتقات الدرجة الثانية (Second-order derivatives)
        - **المبدأ**: قياس معدل تغير gradient (Laplacian)
        - **الكشف**: عن الحواف عند zero-crossings
        - **الحساسية**: أكثر حساسية للضوضاء

        #### 3. خوارزمية Canny (الأكثر شيوعاً)
        - **خطواتها**:
          1. **تنعيم الصورة** (بواسطة Gaussian filter)
          2. **حساب gradient** (عادة باستخدام Sobel)
          3. **قمع غير الأقصى** (Non-maximum suppression)
          4. **التحديد بالعتبة المزدوجة** (Hysteresis thresholding)

        ### مقارنة بين الخوارزميات:

        | الخوارزمية | المميزات | العيوب |
        |------------|----------|--------|
        | **Sobel** | سريعة وبسيطة | حساسة للضوضاء |
        | **Prewitt** | أبسط من Sobel | أقل دقة |
        | **Laplacian** | يكتشف جميع الاتجاهات | حساس جداً للضوضاء |
        | **Canny** | دقيقة وقليلة الضوضاء | بطيئة نوعاً ما |

        ### معاملات مهمة في كشف الحواف:
        - **العتبة المنخفضة**: للحد الأدنى لقوة الحافة
        - **العتبة العالية**: للحد الأدنى لقوة الحافة القوية
        - **حجم النواة**: يؤثر على دقة الكشف
        - **حجم Gaussian**: للتحكم في التنعيم المبدئي
        """)
    
    st.markdown("---")
    
    # التطبيق العملي
    st.subheader("🔍 التجربة العملية: كشف الحواف")
    
    # تحميل الصورة
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📤 تحميل الصورة")
        uploaded_file = st.file_uploader("اختر صورة", type=['jpg', 'jpeg', 'png'], key="edge_upload")
        
        if uploaded_file is not None:
            image = load_image(uploaded_file)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            st.image(image, caption="الصورة الأصلية (رمادية)", use_container_width=True)
        else:
            # إنشاء صورة افتراضية
            image = create_edge_detection_sample()
            st.image(image, caption="الصورة الافتراضية", use_container_width=True)
    
    with col2:
        if image is not None:
            st.markdown("#### ⚙️ اختر خوارزمية كشف الحواف")
            
            edge_method = st.selectbox(
                "الخوارزمية:",
                ["Sobel", "Prewitt", "Laplacian", "Canny", "جميع الطرق"]
            )
            
            if edge_method in ["Sobel", "Prewitt", "Canny"]:
                kernel_size = st.slider("حجم النواة", 3, 7, 3, 2)
                if kernel_size % 2 == 0:
                    kernel_size += 1
            
            if edge_method == "Canny":
                threshold1 = st.slider("العتبة المنخفضة", 0, 255, 100)
                threshold2 = st.slider("العتبة العالية", 0, 255, 200)
            
            if st.button("🔍 تطبيق كشف الحواف"):
                with st.spinner("جاري كشف الحواف..."):
                    if edge_method == "Sobel":
                        result = apply_sobel_edges(image, kernel_size)
                    elif edge_method == "Prewitt":
                        result = apply_prewitt_edges(image)
                    elif edge_method == "Laplacian":
                        result = apply_laplacian_edges(image)
                    elif edge_method == "Canny":
                        result = apply_canny_edges(image, threshold1, threshold2, kernel_size)
                    elif edge_method == "جميع الطرق":
                        result = None  # سيتم التعامل معها بشكل منفصل
                    
                    if edge_method != "جميع الطرق":
                        st.image(result, caption=f"كشف الحواف بـ {edge_method}", use_container_width=True)
                        
                        # إحصائيات عن الحواف
                        edge_pixels = np.sum(result > 0)
                        total_pixels = result.size
                        edge_percentage = (edge_pixels / total_pixels) * 100
                        
                        st.metric("عدد بكسلات الحواف", f"{edge_pixels:,}")
                        st.metric("نسبة الحواف", f"{edge_percentage:.2f}%")
                    else:
                        # عرض جميع الطرق
                        show_all_edge_methods(image)
    
    st.markdown("---")
    
    # قسم إضافي: معالجة متقدمة للحواف
    if image is not None:
        st.subheader("🛠️ معالجة متقدمة للحواف")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📊 تحليل اتجاهات الحواف")
            
            # حساب اتجاهات الحواف باستخدام Sobel
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            
            # حساب magnitude و direction
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            direction = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
            
            st.image(np.uint8(magnitude), caption="قوة الحواف (Magnitude)", use_container_width=True)
        
        with col2:
            st.markdown("##### 🎯 تصفية الحواف حسب القوة")
            
            strength_threshold = st.slider("حد القوة", 0, 255, 50)
            strong_edges = magnitude > strength_threshold
            filtered_edges = np.zeros_like(image)
            filtered_edges[strong_edges] = 255
            
            st.image(filtered_edges, caption="الحواف القوية فقط", use_container_width=True)
            
            # إحصائيات
            strong_count = np.sum(strong_edges)
            st.metric("الحواف القوية", f"{strong_count:,}")
    
    # اختبار قصير
    st.markdown("---")
    with st.expander("🧪 اختبار قصير", expanded=False):
        st.subheader("اختبار فهم كشف الحواف")
        
        q1 = st.radio(
            "1. ما هي أفضل خوارزمية لكشف الحواف من حيث الدقة?",
            ["Sobel", "Prewitt", "Canny", "Laplacian"],
            key="edge_q1"
        )
        
        q2 = st.radio(
            "2. كم عدد الخطوات في خوارزمية Canny?",
            ["3 خطوات", "4 خطوات", "5 خطوات", "6 خطوات"],
            key="edge_q2"
        )
        
        q3 = st.radio(
            "3. ما هو مبدأ عمل مشتقات الدرجة الثانية?",
            ["قياس maximum gradient", "قياس zero-crossings", "استخدام عتبات مزدوجة", "التنعيم Gaussian"],
            key="edge_q3"
        )
        
        if st.button("✅ التحقق من الإجابات", key="edge_check"):
            score = 0
            if q1 == "Canny":
                score += 1
                st.success("السؤال 1: صحيح! Canny هي الأكثر دقة")
            else:
                st.error("السؤال 1: خطأ! Canny هي الأكثر دقة")
            
            if q2 == "4 خطوات":
                score += 1
                st.success("السؤال 2: صحيح! Canny لها 4 خطوات")
            else:
                st.error("السؤال 2: خطأ! Canny لها 4 خطوات")
            
            if q3 == "قياس zero-crossings":
                score += 1
                st.success("السؤال 3: صحيح! مشتقات الدرجة الثانية تبحث عن zero-crossings")
            else:
                st.error("السؤال 3: خطأ! مبدأها هو قياس zero-crossings")
            
            st.info(f"**النتيجة: {score}/3**")
            
            if score == 3:
                st.balloons()
                st.session_state.quiz_score = st.session_state.get('quiz_score', 0) + 3

def create_edge_detection_sample():
    """إنشاء صورة مناسبة لكشف الحواف"""
    image = np.zeros((300, 400), dtype=np.uint8)
    
    # إضافة أشكال مختلفة ذات حواف واضحة
    cv2.rectangle(image, (50, 50), (150, 150), 255, -1)
    cv2.circle(image, (300, 100), 50, 200, -1)
    cv2.rectangle(image, (180, 200), (280, 280), 150, -1)
    
    # إضافة خطوط بأزوايا مختلفة
    cv2.line(image, (200, 50), (250, 200), 255, 2)
    cv2.line(image, (100, 200), (150, 50), 255, 2)
    
    # إضافة نص
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Edges', (120, 250), font, 1, 255, 2, cv2.LINE_AA)
    
    return image

def apply_sobel_edges(image, ksize=3):
    """تطبيق كشف الحواف باستخدام Sobel"""
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.uint8(magnitude)

def apply_prewitt_edges(image):
    """تطبيق كشف الحواف باستخدام Prewitt"""
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    prewitt_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
    prewitt_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
    magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)
    return np.uint8(magnitude)

def apply_laplacian_edges(image):
    """تطبيق كشف الحواف باستخدام Laplacian"""
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return np.uint8(np.absolute(laplacian))

def apply_canny_edges(image, threshold1, threshold2, ksize=3):
    """تطبيق كشف الحواف باستخدام Canny"""
    return cv2.Canny(image, threshold1, threshold2, L2gradient=True)

def show_all_edge_methods(image):
    """عرض جميع طرق كشف الحواف"""
    st.markdown("##### 📊 مقارنة بين جميع خوارزميات كشف الحواف")
    
    methods = {
        "Sobel": apply_sobel_edges(image, 3),
        "Prewitt": apply_prewitt_edges(image),
        "Laplacian": apply_laplacian_edges(image),
        "Canny (100,200)": apply_canny_edges(image, 100, 200)
    }
    
    cols = st.columns(2)
    col_idx = 0
    
    for name, result in methods.items():
        with cols[col_idx]:
            st.image(result, caption=name, use_container_width=True)
            edge_pixels = np.sum(result > 0)
            st.caption(f"بكسلات الحواف: {edge_pixels:,}")
        
        col_idx = (col_idx + 1) % 2
    
    # تحليل مقارن
    st.markdown("##### 📈 تحليل مقارن")
    comparison_data = []
    for name, result in methods.items():
        edge_pixels = np.sum(result > 0)
        comparison_data.append({"الخوارزمية": name, "عدد الحواف": edge_pixels})
    
    st.bar_chart({d["الخوارزمية"]: d["عدد الحواف"] for d in comparison_data})