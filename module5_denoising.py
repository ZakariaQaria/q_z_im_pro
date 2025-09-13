import streamlit as st
import cv2
import numpy as np
from helpers import load_image, add_user_progress, add_noise

def show_module5():
    """عرض المحاضرة الخامسة: إزالة الضوضاء"""
    
    st.header("🔇 المحاضرة 5: إزالة الضوضاء (Denoising)")
    
    # معلومات التقدم
    if st.session_state.progress.get("module5", False):
        st.success("✅ لقد أكملت هذه المحاضرة بالفعل")
    else:
        if add_user_progress("module5"):
            st.balloons()
            st.success("🎉 مبروك! لقد أكملت المحاضرة الخامسة وحصلت على 20 نقطة")
    
    # النظرية
    with st.expander("📖 الشرح النظري", expanded=True):
        st.markdown("""
        ## إزالة الضوضاء من الصور (Image Denoising)

        ### ما هي الضوضاء في الصور؟
        الضوضاء هي variations عشوائية في شدة البكسلات لا تتعلق بالمحتوى الحقيقي للصورة.

        ### أنواع الضوضاء الشائعة:

        #### 1. ضوضاء Gaussian (الطبيعية)
        - **الوصف**: توزيع طبيعي حول القيمة الحقيقية
        - **المظهر**: حبيبات ناعمة منتشرة في الصورة
        - **السبب**: عادة من حساسية الكاميرا أو ظروف الإضاءة الضعيفة

        #### 2. ضوضاء Salt & Pepper (الملح والفلفل)
        - **الوصف**: بكسلات بيضاء وسوداء عشوائية
        - **المظهر**: نقاط بيضاء وسوداء متناثرة
        - **السبب**: أخطاء في نقل البيانات أو تخزينها

        #### 3. ضوضاء Poisson (الكم)
        - **الوصف**: ناتجة عن الطبيعة الكمية للضوء
        - **المظهر**: مشابه لـ Gaussian ولكن يعتمد على شدة الضوء
        - **السبب**: inherent في عملية capture الصورة

        #### 4. ضوضاء Speckle (المرقطة)
        - **الوصف**: ضوضاء مضاعفة ( multiplicative)
        - **المظهر**: حبيبات خشنة
        - **السبب**: في صور الرادار والسونار

        ### طرق إزالة الضوضاء:

        #### 1. Gaussian Filter
        - **المبدأ**: تمويه باستخدام نواة Gaussian
        - **الفعالية**: جيد لضوضاء Gaussian
        - **العيوب**: يسبب blurring للحواف

        #### 2. Median Filter
        - **المبدأ**: استبدال كل بكسل بالوسيط الحسابي للجوار
        - **الفعالية**: ممتاز لضوضاء Salt & Pepper
        - **العيوب**: قد يسبب فقدان التفاصيل الدقيقة

        #### 3. Bilateral Filter
        - **المبدأ**: Gaussian filter يأخذ في الاعتبار تشابه الشدة
        - **الفعالية**: جيد للحفاظ على الحواف while removing noise
        - **العيوب**: أبطأ من الفلاتر التقليدية

        #### 4. Non-Local Means Denoising
        - **المبدأ**: استخدام patches متشابهة من جميع أنحاء الصورة
        - **الفعالية**: فعال جداً للعديد من أنواع الضوضاء
        - **العيوب**: computationally expensive

        #### 5. Wiener Filter
        - **المبدأ**: filter adaptative يعتمد على إحصائيات الضوضاء
        - **الفعالية**: جيد عندما نعرف خصائص الضوضاء

        ### مقاييس جودة إزالة الضوضاء:
        - **PSNR (Peak Signal-to-Noise Ratio)**: يقيس جودة reconstruction
        - **SSIM (Structural Similarity)**: يقيس التشابه البنيوي
        - **MSE (Mean Squared Error)**: متوسط مربعات الفروق
        """)
    
    st.markdown("---")
    
    # التطبيق العملي
    st.subheader("🔧 التجربة العملية: إزالة الضوضاء")
    
    # تحميل الصورة
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📤 تحميل الصورة")
        uploaded_file = st.file_uploader("اختر صورة", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            original_image = load_image(uploaded_file)
            if len(original_image.shape) == 3:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            st.image(original_image, caption="الصورة الأصلية", use_container_width=True)
        else:
            # إنشاء صورة افتراضية
            original_image = create_denoising_sample_image()
            st.image(original_image, caption="الصورة الافتراضية", use_container_width=True)
    
    with col2:
        if original_image is not None:
            st.markdown("#### 🌪️ إضافة ضوضاء للصورة")
            
            noise_type = st.selectbox(
                "نوع الضوضاء:",
                ["Gaussian", "Salt & Pepper", "Poisson", "None"]
            )
            
            if noise_type != "None":
                noise_amount = st.slider("شدة الضوضاء", 0.01, 0.5, 0.1, 0.01)
                
                if st.button("🔄 إضافة الضوضاء"):
                    noisy_image = add_noise(original_image, noise_type.lower(), noise_amount)
                    st.image(noisy_image, caption=f"الصورة مع ضوضاء {noise_type}", use_container_width=True)
                    
                    # حساب مقاييس الجودة
                    mse = np.mean((original_image - noisy_image) ** 2)
                    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
                    
                    st.metric("MSE", f"{mse:.2f}")
                    st.metric("PSNR", f"{psnr:.2f} dB")
            else:
                noisy_image = original_image
            
            st.markdown("#### 🛡️ اختيار طريقة إزالة الضوضاء")
            
            denoise_method = st.selectbox(
                "طريقة الإزالة:",
                ["Gaussian Filter", "Median Filter", "Bilateral Filter", "NLM (Non-Local Means)"]
            )
            
            if st.button("🔧 تطبيق إزالة الضوضاء") and 'noisy_image' in locals():
                with st.spinner("جاري إزالة الضوضاء..."):
                    if denoise_method == "Gaussian Filter":
                        kernel_size = st.slider("حجم النواة", 3, 15, 5, 2)
                        if kernel_size % 2 == 0:
                            kernel_size += 1
                        denoised = cv2.GaussianBlur(noisy_image, (kernel_size, kernel_size), 0)
                    
                    elif denoise_method == "Median Filter":
                        kernel_size = st.slider("حجم النواة", 3, 15, 5, 2)
                        if kernel_size % 2 == 0:
                            kernel_size += 1
                        denoised = cv2.medianBlur(noisy_image, kernel_size)
                    
                    elif denoise_method == "Bilateral Filter":
                        d = st.slider("قطر الجوار", 1, 15, 5, 2)
                        sigma_color = st.slider("Sigma Color", 1, 100, 75)
                        sigma_space = st.slider("Sigma Space", 1, 100, 75)
                        denoised = cv2.bilateralFilter(noisy_image, d, sigma_color, sigma_space)
                    
                    elif denoise_method == "NLM (Non-Local Means)":
                        h = st.slider("معلمة القوة (h)", 1, 30, 10)
                        denoised = cv2.fastNlMeansDenoising(noisy_image, None, h, 7, 21)
                    
                    st.image(denoised, caption=f"بعد إزالة الضوضاء ({denoise_method})", use_container_width=True)
                    
                    # حساب مقاييس الجودة بعد الإزالة
                    if noise_type != "None":
                        mse_after = np.mean((original_image - denoised) ** 2)
                        psnr_after = 20 * np.log10(255.0 / np.sqrt(mse_after)) if mse_after > 0 else float('inf')
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("MSE بعد", f"{mse_after:.2f}", f"{mse - mse_after:.2f}")
                        with col2:
                            st.metric("PSNR بعد", f"{psnr_after:.2f} dB", f"{psnr_after - psnr:.2f}")
    
    st.markdown("---")
    
    # قسم إضافي: مقارنة بين طرق الإزالة
    st.subheader("📊 مقارنة بين طرق إزالة الضوضاء")
    
    if original_image is not None and 'noisy_image' in locals():
        st.markdown("##### مقارنة بين الطرق المختلفة (لضوضاء Gaussian):")
        
        # إنشاء نسخة noisy إذا لم تكن موجودة
        if 'noisy_image' not in locals():
            noisy_image = add_noise(original_image, "gaussian", 0.1)
        
        # تطبيق طرق إزالة مختلفة
        methods = {
            "Gaussian 5x5": cv2.GaussianBlur(noisy_image, (5, 5), 0),
            "Median 5x5": cv2.medianBlur(noisy_image, 5),
            "Bilateral": cv2.bilateralFilter(noisy_image, 9, 75, 75),
            "NLM": cv2.fastNlMeansDenoising(noisy_image, None, 10, 7, 21)
        }
        
        cols = st.columns(4)
        for (name, result), col in zip(methods.items(), cols):
            with col:
                st.image(result, caption=name, use_container_width=True)
                # حساب PSNR
                mse = np.mean((original_image - result) ** 2)
                psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
                st.caption(f"PSNR: {psnr:.2f} dB")
    
    # اختبار قصير
    st.markdown("---")
    with st.expander("🧪 اختبار قصير", expanded=False):
        st.subheader("اختبار فهم إزالة الضوضاء")
        
        q1 = st.radio(
            "1. أي نوع من الفلاتر أفضل لضوضاء Salt & Pepper?",
            ["Gaussian Filter", "Median Filter", "Bilateral Filter", "Wiener Filter"]
        )
        
        q2 = st.radio(
            "2. ما هو مقياس PSNR?",
            ["مقياس للضوضاء", "نسبة الإشارة إلى الضوضاء", "مقياس للتباين", "مقياس للألوان"]
        )
        
        q3 = st.radio(
            "3. ما هي ميزة Bilateral Filter مقارنة بـ Gaussian Filter?",
            ["أسرع", "يحافظ على الحواف", "أفضل للضوضاء الملونة", "أسهل في الضبط"]
        )
        
        if st.button("✅ التحقق من الإجابات"):
            score = 0
            if q1 == "Median Filter":
                score += 1
                st.success("السؤال 1: صحيح! Median Filter أفضل لضوضاء Salt & Pepper")
            else:
                st.error("السؤال 1: خطأ! Median Filter هو الأفضل لهذا النوع")
            
            if q2 == "نسبة الإشارة إلى الضوضاء":
                score += 1
                st.success("السؤال 2: صحيح! PSNR = Peak Signal-to-Noise Ratio")
            else:
                st.error("السؤال 2: خطأ! PSNR هي نسبة الإشارة إلى الضوضاء")
            
            if q3 == "يحافظ على الحواف":
                score += 1
                st.success("السؤال 3: صحيح! Bilateral Filter يحافظ على الحواف أثناء إزالة الضوضاء")
            else:
                st.error("السؤال 3: خطأ! ميزة Bilateral Filter هي الحفاظ على الحواف")
            
            st.info(f"**النتيجة: {score}/3**")
            
            if score == 3:
                st.balloons()
                st.session_state.quiz_score = st.session_state.get('quiz_score', 0) + 3

def create_denoising_sample_image():
    """إنشاء صورة مناسبة لتجارب إزالة الضوضاء"""
    image = np.zeros((300, 400), dtype=np.uint8)
    
    # إضافة تدرج
    for i in range(400):
        intensity = int(255 * i / 400)
        image[:, i] = intensity
    
    # إضافة تفاصيل دقيقة
    cv2.rectangle(image, (50, 50), (150, 150), 100, -1)
    cv2.circle(image, (300, 100), 50, 200, -1)
    
    # إضافة نص صغير
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Text', (100, 250), font, 0.7, 255, 2, cv2.LINE_AA)
    
    # إضافة حواف رفيعة
    for i in range(10):
        cv2.line(image, (200 + i, 50), (250 + i, 200), 150, 1)
    
    return image