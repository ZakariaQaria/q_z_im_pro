import streamlit as st
import cv2
import numpy as np
import os
from helpers import load_image, add_user_progress
from achievements import add_achievement

def show_module9():
    """عرض المحاضرة التاسعة: المشروع الختامي"""
    
    st.header("🎓 المحاضرة 9: المشروع الختامي")
    
    # معلومات التقدم
    if st.session_state.progress.get("module9", False):
        st.success("✅ لقد أكملت هذه المحاضرة بالفعل")
    else:
        if add_user_progress("module9"):
            st.balloons()
            st.success("🎉 مبروك! لقد أكملت المحاضرة التاسعة وحصلت على 20 نقطة")
            add_achievement("خبير معالجة الصور", "إكمال جميع محاضرات الدورة")
    
    # النظرية
    with st.expander("📖 نظرة عامة", expanded=True):
        st.markdown("""
        ## المشروع الختامي: تطبيق عملي متكامل
        
        في هذه المحاضرة، ستطبق كل ما تعلمته في المحاضرات السابقة لبناء pipeline كامل لمعالجة الصور.
        
        ### 🎯 الميزات المتوفرة:
        - ✅ رفع الصور من الجهاز
        - ✅ اختيار سلسلة عمليات متكاملة
        - ✅ معالجة الصورة خطوة بخطوة
        - ✅ عرض النتائج بمقارنة قبل/بعد
        - ✅ حفظ الصورة الناتجة
        - ✅ تحميل الملف المعالج
        """)
    
    st.markdown("---")
    
    # المشروع العملي
    st.subheader("🔧 المشروع العملي: بناء Pipeline متكامل")
    
    # تحميل الصورة
    st.markdown("#### 📤 الخطوة 1: تحميل الصورة")
    uploaded_file = st.file_uploader("اختر صورة للمعالجة", type=['jpg', 'jpeg', 'png'], key="final_upload")
    
    if uploaded_file is not None:
        original_image = load_image(uploaded_file)
        st.image(original_image, caption="الصورة الأصلية", use_container_width=True)
    else:
        # إنشاء صورة افتراضية
        original_image = create_final_project_sample()
        st.image(original_image, caption="الصورة الافتراضية", use_container_width=True)
    
    if original_image is not None:
        st.markdown("---")
        st.markdown("#### ⚙️ الخطوة 2: بناء Pipeline المعالجة")
        
        # أمثلة على سلاسل العمليات الجاهزة
        st.markdown("##### 🎯 أمثلة جاهزة:")
        
        preset_options = {
            "تدرج رمادي + تمويه + حواف": ["convert_grayscale", "remove_noise", "detect_edges"],
            "تحسين تباين + كشف حواف": ["adjust_contrast", "detect_edges"],
            "إزالة ضوضاء + تحسين": ["remove_noise", "adjust_contrast"],
            "كشف حواف متقدم": ["convert_grayscale", "remove_noise", "detect_edges", "apply_morphology"]
        }
        
        selected_preset = st.selectbox("اختر pipeline جاهز:", list(preset_options.keys()))
        
        # تطبيق الإعدادات الجاهزة
        if selected_preset:
            preset_settings = preset_options[selected_preset]
            convert_grayscale = "convert_grayscale" in preset_settings
            adjust_contrast = "adjust_contrast" in preset_settings
            remove_noise = "remove_noise" in preset_settings
            detect_edges = "detect_edges" in preset_settings
            apply_morphology = "apply_morphology" in preset_settings
        else:
            convert_grayscale = False
            adjust_contrast = False
            remove_noise = False
            detect_edges = False
            apply_morphology = False
        
        st.markdown("##### ⚙️ تعديل إعدادات مخصصة:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            convert_grayscale = st.checkbox("تحويل إلى تدرج رمادي", value=convert_grayscale)
            adjust_contrast = st.checkbox("ضبط التباين", value=adjust_contrast)
            remove_noise = st.checkbox("إزالة الضوضاء", value=remove_noise)
        
        with col2:
            detect_edges = st.checkbox("كشف الحواف", value=detect_edges)
            apply_morphology = st.checkbox("عمليات مورفولوجية", value=apply_morphology)
            add_overlay = st.checkbox("إضافة overlay على الحواف", value=False)
        
        # معاملات التحكم
        st.markdown("##### ⚖️ ضبط المعاملات:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            contrast_level = st.slider("مستوى التباين", 0.5, 3.0, 1.2, 0.1)
            noise_level = st.slider("قوة إزالة الضوضاء", 1, 15, 5)
        
        with col2:
            edge_threshold = st.slider("عتبة الحواف", 50, 200, 100)
            morph_size = st.slider("حجم النواة", 1, 10, 3)
        
        with col3:
            overlay_opacity = st.slider("شفافية الـ Overlay", 0.0, 1.0, 0.5, 0.1)
            transform_angle = st.slider("زاوية الدوران", -30, 30, 0)
        
        # زر التشغيل الرئيسي
        if st.button("🚀 تشغيل Pipeline", type="primary", use_container_width=True):
            with st.spinner("جاري معالجة الصورة..."):
                # تطبيق خطوات المعالجة
                processed_image = original_image.copy()
                steps_log = []
                intermediate_results = []
                
                # 1. تحويل إلى تدرج رمادي
                if convert_grayscale:
                    if len(processed_image.shape) == 3:
                        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
                    steps_log.append("✅ تحويل إلى تدرج رمادي")
                    intermediate_results.append(("تدرج رمادي", processed_image.copy()))
                
                # 2. ضبط التباين
                if adjust_contrast:
                    if len(processed_image.shape) == 2:
                        processed_image = cv2.convertScaleAbs(processed_image, alpha=contrast_level, beta=0)
                    else:
                        processed_image = cv2.convertScaleAbs(processed_image, alpha=contrast_level, beta=0)
                    steps_log.append(f"✅ ضبط التباين (مستوى: {contrast_level})")
                    intermediate_results.append(("بعد التباين", processed_image.copy()))
                
                # 3. إزالة الضوضاء - الإصلاح هنا
                if remove_noise:
                    # تأكد أن حجم kernel فردي
                    kernel_size = noise_level
                    if kernel_size % 2 == 0:  # إذا كان زوجي
                        kernel_size += 1      # اجعله فردي
                    
                    processed_image = cv2.medianBlur(processed_image, kernel_size)
                    steps_log.append(f"✅ إزالة الضوضاء (قوة: {kernel_size})")
                    intermediate_results.append(("بعد إزالة الضوضاء", processed_image.copy()))
                
                # 4. كشف الحواف
                edges = None
                if detect_edges:
                    if len(processed_image.shape) == 3:
                        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = processed_image
                    edges = cv2.Canny(gray, edge_threshold, edge_threshold * 2)
                    steps_log.append(f"✅ كشف الحواف (عتبة: {edge_threshold})")
                    intermediate_results.append(("كشف الحواف", edges.copy()))
                
                # 5. العمليات المورفولوجية
                if apply_morphology and edges is not None:
                    kernel = np.ones((morph_size, morph_size), np.uint8)
                    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
                    steps_log.append(f"✅ عمليات مورفولوجية (حجم: {morph_size}x{morph_size})")
                    intermediate_results.append(("بعد العمليات المورفولوجية", edges.copy()))
                
                # 6. إضافة overlay للحواف
                result_image = original_image.copy()
                if add_overlay and edges is not None:
                    if len(result_image.shape) == 2:
                        result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
                    
                    # إنشاء overlay أحمر للحواف
                    overlay = result_image.copy()
                    overlay[edges > 0] = [0, 0, 255]  # أحمر
                    result_image = cv2.addWeighted(overlay, overlay_opacity, result_image, 1 - overlay_opacity, 0)
                    steps_log.append(f"✅ إضافة overlay للحواف (شفافية: {overlay_opacity})")
                    intermediate_results.append(("بعد إضافة Overlay", result_image.copy()))
                
                # عرض النتائج
                st.markdown("---")
                st.markdown("#### 📊 النتائج النهائية")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(original_image, caption="الصورة الأصلية", use_container_width=True)
                
                with col2:
                    if 'result_image' in locals() and result_image is not None:
                        st.image(result_image, caption="الصورة المعالجة النهائية", use_container_width=True)
                    elif edges is not None:
                        st.image(edges, caption="الحواف المكتشفة", use_container_width=True)
                    else:
                        st.image(processed_image, caption="الصورة المعالجة", use_container_width=True)
                
                # عرض الخطوات المتوسطة
                if len(intermediate_results) > 1:
                    st.markdown("#### 🔍 الخطوات المتوسطة")
                    cols = st.columns(len(intermediate_results))
                    for idx, (step_name, step_image) in enumerate(intermediate_results):
                        with cols[idx]:
                            st.image(step_image, caption=step_name, use_container_width=True)
                
                # عرض سجل الخطوات
                st.markdown("#### 📝 سجل خطوات المعالجة:")
                for step in steps_log:
                    st.write(f"• {step}")
                
                # إحصائيات
                st.markdown("#### 📈 إحصائيات المعالجة:")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if edges is not None:
                        edge_pixels = np.sum(edges > 0)
                        st.metric("بكسلات الحواف", f"{edge_pixels:,}")
                
                with col2:
                    processing_time = len(steps_log) * 0.3
                    st.metric("الوقت التقديري", f"{processing_time:.1f} ثانية")
                
                with col3:
                    if 'result_image' in locals():
                        img_size = result_image.nbytes / 1024
                    else:
                        img_size = processed_image.nbytes / 1024
                    st.metric("حجم الصورة", f"{img_size:.1f} كيلوبايت")
                
                # زر حفظ الصورة الناتجة
                st.markdown("---")
                st.markdown("#### 💾 حفظ النتيجة")
                
                # إنشاء زر الحفظ
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("💾 حفظ الصورة الناتجة", type="secondary", use_container_width=True):
                        try:
                            # تحديد الصورة للحفظ
                            if 'result_image' in locals() and result_image is not None:
                                image_to_save = result_image
                            elif edges is not None:
                                image_to_save = edges
                            else:
                                image_to_save = processed_image
                            
                            # تحويل الألوان إذا needed
                            if len(image_to_save.shape) == 3:
                                image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
                            
                            # حفظ الصورة
                            success = cv2.imwrite("processed_image.jpg", image_to_save)
                            
                            if success:
                                st.success("✅ تم حفظ الصورة بنجاح!")
                                st.info("📁 الملف: processed_image.jpg")
                                
                                # عرض صورة مصغرة
                                try:
                                    saved_image = cv2.imread("processed_image.jpg")
                                    if saved_image is not None:
                                        if len(saved_image.shape) == 3:
                                            saved_image = cv2.cvtColor(saved_image, cv2.COLOR_BGR2RGB)
                                        st.image(saved_image, caption="الصورة المحفوظة", width=300)
                                except Exception as e:
                                    st.warning("⚠️ لا يمكن عرض الصورة المحفوظة")
                            else:
                                st.error("❌ فشل في حفظ الصورة")
                                
                        except Exception as e:
                            st.error(f"❌ خطأ في الحفظ: {str(e)}")
                
                with col2:
                    # زر تحميل إضافي
                    if os.path.exists("processed_image.jpg"):
                        with open("processed_image.jpg", "rb") as file:
                            st.download_button(
                                label="⬇️ تحميل الملف",
                                data=file,
                                file_name="صورتي_المعالجة.jpg",
                                mime="image/jpeg",
                                use_container_width=True
                            )
    
    st.markdown("---")
    
    # قسم التحدي الإبداعي
    st.subheader("🎨 تصميم Pipeline مخصص")
    
    st.markdown("""
    ### 🚀 صمم pipeline خاص بك:
    - اختار التقنيات المناسبة لمشروعك
    - ضبط المعاملات للحصول على أفضل نتيجة
    - احفظ النتائج وشاركها
    """)

def create_final_project_sample():
    """إنشاء صورة افتراضية للمشروع"""
    image = np.zeros((400, 500, 3), dtype=np.uint8)
    
    # خلفية متدرجة
    for i in range(500):
        color = int(255 * i / 500)
        image[:, i] = [color, color//2, 255-color]
    
    # إضافة أشكال مختلفة
    cv2.rectangle(image, (50, 50), (150, 150), (0, 0, 255), -1)  # أحمر
    cv2.circle(image, (400, 100), 60, (0, 255, 0), -1)  # أخضر
    cv2.rectangle(image, (250, 250), (350, 350), (255, 0, 0), -1)  # أزرق
    
    # إضافة نص
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'مشروع معالجة الصور', (100, 380), font, 1, (255, 255, 255), 3, cv2.LINE_AA)
    
    return image