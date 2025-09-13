import streamlit as st
import cv2
import numpy as np
from helpers import load_image
from datetime import datetime

def show_evaluation():
    """نظام تقييم أداء المستخدم"""
    
    st.header("📊 تقييم الأداء")
    
    if 'user_xp' not in st.session_state:
        st.session_state.user_xp = 0
    if 'user_level' not in st.session_state:
        st.session_state.user_level = 1
    
    # إحصائيات المستخدم
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("المستوى", st.session_state.user_level)
    
    with col2:
        st.metric("النقاط", st.session_state.user_xp)
    
    with col3:
        progress = sum(st.session_state.progress.values())
        st.metric("المحاضرات المكتملة", f"{progress}/9")
    
    # شريط التقدم
    xp_needed = st.session_state.user_level * 100
    xp_progress = min(st.session_state.user_xp / xp_needed, 1.0)
    
    st.progress(xp_progress)
    st.caption(f"التقدم للlevel {st.session_state.user_level + 1}: {st.session_state.user_xp}/{xp_needed}")
    
    # المهارات المكتسبة
    st.markdown("### 🎯 المهارات المكتسبة")
    
    skills = {
        "أساسيات الصور الرقمية": st.session_state.progress.get("module1", False),
        "أنظمة الألوان": st.session_state.progress.get("module2", False),
        "العمليات على البكسل": st.session_state.progress.get("module3", False),
        "الفلاتر والالتفاف": st.session_state.progress.get("module4", False),
        "إزالة الضوضاء": st.session_state.progress.get("module5", False),
        "كشف الحواف": st.session_state.progress.get("module6", False),
        "العمليات المورفولوجية": st.session_state.progress.get("module7", False),
        "التحويلات الهندسية": st.session_state.progress.get("module8", False),
        "المشاريع المتكاملة": st.session_state.progress.get("module9", False)
    }
    
    for skill, acquired in skills.items():
        status = "✅" if acquired else "❌"
        st.write(f"{status} {skill}")
    
    # التوصيات
    st.markdown("### 💡 التوصيات")
    
    recommendations = []
    if not st.session_state.progress.get("module1", False):
        recommendations.append("ابدأ بالمحاضرة الأولى: مدخل إلى الصور الرقمية")
    
    if st.session_state.progress.get("module1", False) and not st.session_state.progress.get("module2", False):
        recommendations.append("انتقل إلى المحاضرة الثانية: أنظمة الألوان")
    
    if st.session_state.user_xp < 100:
        recommendations.append("أكمل المزيد من المحاضرات لكسب النقاط")
    
    if not recommendations:
        recommendations.append("أداء ممتاز! استمر في التعلم")
    
    for rec in recommendations:
        st.info(f"📌 {rec}")
    
    # تحميل الشهادة
    st.markdown("---")
    st.markdown("### 📜 الشهادة الإلكترونية")
    
    progress_count = sum(st.session_state.progress.values())
    if st.button("📄 تحميل شهادة الإنجاز", type="primary"):
        if progress_count >= 1:  # غير لـ 1 لأجل التجربة
            generate_certificate()
        else:
            st.warning("⚠️ تحتاج إلى إكمال محاضرات أولاً لتحميل الشهادة")

def generate_certificate():
    """إنشاء شهادة إنجاز"""
    st.success("🎓 تم إنشاء شهادة الإنجاز بنجاح!")
    
    # إنشاء بيانات الشهادة
    certificate_data = create_certificate_image()
    
    # زر التحميل
    st.download_button(
        label="⬇️ تحميل الشهادة",
        data=certificate_data,
        file_name="شهادة_معالجة_الصور.png",
        mime="image/png",
        use_container_width=True
    )

def create_certificate_image():
    """إنشاء صورة الشهادة"""
    # إنشاء صورة الشهادة
    width, height = 800, 600
    certificate = np.ones((height, width, 3), dtype=np.uint8) * 255  # خلفية بيضاء
    
    # إضافة إطار ذهبي
    cv2.rectangle(certificate, (20, 20), (width-20, height-20), (0, 165, 255), 8)
    
    # إضافة عنوان
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    cv2.putText(certificate, "شهادة إنجاز", (250, 100), font, 2, (0, 0, 0), 3)
    
    # إضافة اسم المستخدم
    user_name = st.session_state.get('user_name', 'المتعلم')
    cv2.putText(certificate, f"يُمنح هذه الشهادة إلى: {user_name}", 
                (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # إضافة التفاصيل
    details = [
        "لإكمال دورة معالجة الصور التفاعلية",
        f"المستوى: {st.session_state.get('user_level', 1)}",
        f"النقاط: {st.session_state.get('user_xp', 0)}",
        f"المحاضرات المكتملة: {sum(st.session_state.get('progress', {}).values())}/9"
    ]
    
    y_position = 250
    for detail in details:
        cv2.putText(certificate, detail, (200, y_position), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_position += 40
    
    # إضافة تاريخ
    date = datetime.now().strftime("%Y-%m-%d")
    cv2.putText(certificate, f"التاريخ: {date}", (300, 450), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # إضافة توقيع
    cv2.putText(certificate, "التوقيع: ___________", (300, 500), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # تحويل إلى bytes للتحميل
    _, buffer = cv2.imencode('.png', certificate)
    return buffer.tobytes()

def calculate_overall_score():
    """حساب النتيجة الإجمالية"""
    progress_score = sum(st.session_state.progress.values()) * 10
    quiz_score = st.session_state.get('quiz_score', 0)
    challenges_score = st.session_state.get('challenge_points', 0)
    
    return progress_score + quiz_score + challenges_score

def get_performance_level():
    """الحصول على مستوى الأداء"""
    total_score = calculate_overall_score()
    
    if total_score >= 200:
        return "ممتاز 🎯"
    elif total_score >= 150:
        return "جيد جداً ⭐"
    elif total_score >= 100:
        return "جيد 👍"
    else:
        return "مبتدئ 📚"