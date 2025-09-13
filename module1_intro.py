import streamlit as st
import cv2
import numpy as np
from PIL import Image
from helpers import load_image, get_image_info, add_user_progress
def show_module1():
    """عرض المحاضرة الأولى: مدخل ومعمارية الصور الرقمية"""
    
    st.header("📊 المحاضرة 1: مدخل ومعمارية الصور الرقمية")
    
    # معلومات التقدم
    if st.session_state.progress.get("module1", False):
        st.success("✅ لقد أكملت هذه المحاضرة بالفعل")
    else:
        if add_user_progress("module1"):
            st.balloons()
            st.success("🎉 مبروك! لقد أكملت المحاضرة الأولى وحصلت على 20 نقطة")
    
    # النظرية
    with st.expander("📖 الشرح النظري", expanded=True):
        st.markdown("""
        ### ما هي الصورة الرقمية؟
        
        الصورة الرقمية هي تمثيل رقمي للصورة المرئية، تتكون من مصفوفة ثنائية الأبعاد من البكسلات (pixels). 
        كل بكسل يمثل نقطة في الصورة ويحمل معلومات اللون والإضاءة في تلك النقطة.
        
        ### مكونات الصورة الرقمية:
        
        1. **البكسل (Pixel)**: 
           - أصغر وحدة في الصورة الرقمية
           - يحمل لوناً وقيمة إضاءة
           - المزيد من البكسلات يعني دقة أعلى للصورة
        
        2. **الأبعاد (Dimensions)**: 
           - عدد البكسلات في العرض والارتفاع (مثلاً 640×480)
           - تحدد دقة ووضوح الصورة
        
        3. **القنوات (Channels)**: 
           - مكونات الألوان التي تشكل الصورة
           - RGB: الأحمر، الأخضر، الأزرق (3 قنوات)
           - Grayscale: التدرج الرمادي (قناة واحدة)
        
        4. **العمق اللوني (Bit Depth)**: 
           - عدد البتات المستخدمة لتمثيل لون كل بكسل
           - 8-bit: 256 لون (0-255)
           - 16-bit: 65,536 لون
           - 24-bit: 16.7 مليون لون (8 بت لكل قناة)
        
        ### أنواع الصور الرقمية:
        
        - **صورة ثنائية (Binary)**: بكسل إما أبيض (1) أو أسود (0)
        - **صورة رمادية (Grayscale)**: تدرج بين الأسود والأبيض (0-255)
        - **صورة ملونة (Color)**: تحتوي على معلومات الألوان (عادة RGB)
        - **صورة متعددة القنوات (Multichannel)**: تحتوي على أكثر من 3 قنوات
        """)
    
    st.markdown("---")
    
    # التطبيق العملي
    st.subheader("🔍 التجربة العملية")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📤 تحميل الصورة")
        option = st.radio("اختر مصدر الصورة:", 
                         ["استخدام صورة افتراضية", "رفع صورة"],
                         horizontal=True)
        
        if option == "استخدام صورة افتراضية":
            # إنشاء صورة افتراضية
            image = create_sample_image()
            st.image(image, caption="الصورة الافتراضية",  use_container_width=True)
        else:
            uploaded_file = st.file_uploader("اختر صورة لتحميلها", 
                                           type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'])
            if uploaded_file is not None:
                image = load_image(uploaded_file)
                st.image(image, caption="الصورة المرفوعة", use_container_width=True)
            else:
                image = create_sample_image()
                st.image(image, caption="الصورة الافتراضية (حتى تقوم برفع صورة)", use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 معلومات الصورة")
        if image is not None:
            st.markdown(get_image_info(image))
            
            # عرض إحصائيات إضافية
            if len(image.shape) > 2:
                st.markdown("#### 🎨 إحصائيات القنوات اللونية")
                
                channels = cv2.split(image)
                channel_names = ["أحمر", "أخضر", "أزرق"]
                
                for i, channel in enumerate(channels):
                    if i < len(channel_names):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"متوسط {channel_names[i]}", f"{np.mean(channel):.1f}")
                        with col2:
                            st.metric(f"أعلى {channel_names[i]}", f"{np.max(channel)}")
                        with col3:
                            st.metric(f"أدنى {channel_names[i]}", f"{np.min(channel)}")
            
            # عرض القنوات اللونية بشكل منفصل
            if len(image.shape) > 2 and image.shape[2] >= 3:
                st.markdown("#### 🎭 القنوات اللونية المنفصلة")
                
                channels = cv2.split(image)
                channel_names = ["قناة الأحمر", "قناة الأخضر", "قناة الأزرق"]
                
                cols = st.columns(3)
                for i, col in enumerate(cols):
                    if i < len(channel_names):
                        # إنشاء صورة للقناة مع تلوينها
                        channel_img = np.zeros_like(image)
                        channel_img[:,:,i] = channels[i]
                        col.image(channel_img, caption=channel_names[i], use_container_width=True)
        else:
            st.warning("⚠️ يرجى تحميل صورة أو اختيار صورة افتراضية")
    
    # أداة تفاعلية إضافية
    st.markdown("---")
    st.markdown("#### 🔬 أداة فحص البكسلات")
    
    if image is not None:
        height, width = image.shape[:2]
        
        col1, col2 = st.columns(2)
        with col1:
            x_pos = st.slider("موضع أفقي (X)", 0, width-1, width//2)
        with col2:
            y_pos = st.slider("موضع عمودي (Y)", 0, height-1, height//2)
        
        # عرض معلومات البكسل المحدد
        if len(image.shape) == 2:
    # تأكد أن هنا في مسافات بادئة (Tab أو 4 spaces)
          st.image(image, caption="الصورة الأصلية", use_container_width=True)
    else:
    # وهنا أيضاً مسافات بادئة
       st.image(image, caption="الصورة الأصلية", use_container_width=True)
def create_sample_image():
    """إنشاء صورة افتراضية للعرض"""
    # إنشاء صورة ذات خلفية متدرجة
    width, height = 400, 300
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # إضافة تدرج لوني
    for i in range(width):
        color = int(255 * i / width)
        image[:, i] = [color, color//2, 255-color]
    
    # إضافة أشكال هندسية
    cv2.rectangle(image, (50, 50), (150, 150), (0, 255, 0), -1)
    cv2.circle(image, (300, 100), 50, (255, 0, 0), -1)
    cv2.line(image, (200, 50), (250, 200), (0, 0, 255), 3)
    
    # إضافة نص
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'OpenCV', (100, 250), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return image