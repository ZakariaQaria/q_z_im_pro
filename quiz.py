import streamlit as st
import random
from achievements import add_achievement

def show_quiz():
    """عرض الاختبار النهائي"""
    
    st.header("📝 الاختبار النهائي - معالجة الصور")
    st.markdown("اختبر معرفتك الكاملة في معالجة الصور من خلال هذا الاختبار الشامل.")
    
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
        st.session_state.quiz_score = 0
        st.session_state.quiz_answers = {}
    
    # الأسئلة مع الإجابات الصحيحة
    questions = [
        {
            "id": 1,
            "question": "ما هو البكسل في الصورة الرقمية؟",
            "options": [
                "أصغر وحدة في الصورة الرقمية",
                "نوع من أنواع الفلاتر",
                "أداة لقياس جودة الصورة",
                "برنامج لمعالجة الصور"
            ],
            "correct": 0,
            "points": 2
        },
        {
            "id": 2,
            "question": "كم قناة لونية في نظام الألوان RGB؟",
            "options": ["1", "2", "3", "4"],
            "correct": 2,
            "points": 2
        },
        {
            "id": 3,
            "question": "أي من هذه العمليات تستخدم لكشف الحواف؟",
            "options": [
                "التحويل إلى الرمادي",
                "تطبيق فلتر Gaussian",
                "كشف Canny",
                "العمليات المورفولوجية"
            ],
            "correct": 2,
            "points": 3
        },
        {
            "id": 4,
            "question": "ما هو الغرض من العمليات المورفولوجية؟",
            "options": [
                "تحسين ألوان الصورة",
                "معالجة أشكال الأجسام في الصورة الثنائية",
                "إضافة تأثيرات فنية للصورة",
                "زيادة دقة الصورة"
            ],
            "correct": 1,
            "points": 3
        },
        {
            "id": 5,
            "question": "ما الفرق بين التآكل والتمدد في العمليات المورفولوجية؟",
            "options": [
                "التآكل يوسع الأشياء والتمدد يضيقها",
                "التمدد يوسع الأشياء والتآكل يضيقها",
                "كلاهما لهما نفس التأثير",
                "التآكل للصور الملونة والتمدد للرمادية"
            ],
            "correct": 1,
            "points": 4
        },
        {
            "id": 6,
            "question": "أي من هذه الخوارزميات تعتبر الأفضل لكشف الحواف؟",
            "options": ["Sobel", "Prewitt", "Laplacian", "Canny"],
            "correct": 3,
            "points": 3
        },
        {
            "id": 7,
            "question": "ما هو نظام الألوان HSV؟",
            "options": [
                "نظام ألوان للطباعة",
                "نظام ألوان يعتمد على الصبغة والإشباع والقيمة",
                "نظام ألوان للصور الطبية",
                "نظام ألوان قديم"
            ],
            "correct": 1,
            "points": 3
        },
        {
            "id": 8,
            "question": "ما هو التحويل الأفيني (Affine Transform)؟",
            "options": [
                "تحويل يحافظ على الخطوط المتوازية",
                "تحويل يحافظ على الزوايا فقط",
                "تحويل للصور الملونة فقط",
                "تحويل للصور الرمادية فقط"
            ],
            "correct": 0,
            "points": 4
        },
        {
            "id": 9,
            "question": "ما هو الغرض من إزالة الضوضاء في الصور؟",
            "options": [
                "زيادة حجم الصورة",
                "تحسين جودة الصورة وإزالة التشويش",
                "تغيير ألوان الصورة",
                "إضافة تأثيرات فنية"
            ],
            "correct": 1,
            "points": 2
        },
        {
            "id": 10,
            "question": "ما هي طريقة Bicubic Interpolation؟",
            "options": [
                "أسرع طريقة للاستيفاء",
                "أدق طريقة للاستيفاء ولكنها بطيئة",
                "طريقة للكشف عن الحواف",
                "طريقة للتحويل إلى الرمادي"
            ],
            "correct": 1,
            "points": 4
        }
    ]
    
    if not st.session_state.quiz_submitted:
        with st.form("quiz_form"):
            st.subheader("الاختبار النهائي - 10 أسئلة")
            st.info("اختر الإجابة الصحيحة لكل سؤال. الإجمالي: 30 نقطة")
            
            user_answers = {}
            
            for i, q in enumerate(questions):
                st.markdown(f"**السؤال {i+1}: {q['question']}** ({q['points']} نقاط)")
                
                # عرض الخيارات بشكل عشوائي
                options = q['options']
                random_seed = q['id'] + st.session_state.get('user_id', 0)
                random.Random(random_seed).shuffle(options)
                
                user_answers[q['id']] = st.radio(
                    f"الاختيارات_{q['id']}",
                    options,
                    key=f"q_{q['id']}",
                    label_visibility="collapsed",
                    index=None
                )
                st.markdown("---")
            
            submitted = st.form_submit_button("📤 تقديم الإجابات")
            
            if submitted:
                # التحقق من إجابة جميع الأسئلة
                if any(answer is None for answer in user_answers.values()):
                    st.error("⚠️ يرجى الإجابة على جميع الأسئلة")
                    return
                
                st.session_state.quiz_submitted = True
                st.session_state.quiz_answers = user_answers
                
                # تصحيح الإجابات
                score = 0
                correct_answers = 0
                
                for q in questions:
                    user_answer = user_answers[q['id']]
                    correct_answer = q['options'][q['correct']]
                    
                    if user_answer == correct_answer:
                        score += q['points']
                        correct_answers += 1
                
                st.session_state.quiz_score = score
                
                # عرض النتائج
                st.success(f"**النتيجة: {score}/30**")
                st.progress(score / 30)
                
                # تقييم الأداء
                if score >= 25:
                    st.balloons()
                    st.success("🎉 ممتاز! أداء رائع")
                    add_achievement("خبير معالجة الصور", "الحصول على درجة ممتازة في الاختبار")
                elif score >= 20:
                    st.success("👏 جيد جداً!")
                elif score >= 15:
                    st.info("👍 جيد")
                else:
                    st.warning("📚 تحتاج إلى مزيد من الدراسة")
                
                # عرض الإجابات الصحيحة
                with st.expander("📋 عرض الإجابات الصحيحة", expanded=False):
                    for q in questions:
                        st.markdown(f"**السؤال {q['id']}:** {q['question']}")
                        st.markdown(f"**الإجابة الصحيحة:** {q['options'][q['correct']]}")
                        st.markdown("---")
                
                # تحديث نقاط المستخدم
                st.session_state.user_xp += score
                if score >= 24:
                    add_achievement("سيد الاختبارات", "الحصول على 24+ نقطة في الاختبار النهائي")
    
    else:
        # عرض النتائج بعد التقديم
        st.success(f"**النتيجة النهائية: {st.session_state.quiz_score}/30**")
        
        if st.button("🔄 إعادة الاختبار"):
            st.session_state.quiz_submitted = False
            st.session_state.quiz_score = 0
            st.session_state.quiz_answers = {}
            st.rerun()

def get_quiz_score():
    """الحصول على نتيجة الاختبار"""
    return st.session_state.get('quiz_score', 0)

def is_quiz_passed():
    """التحقق إذا تم اجتياز الاختبار"""
    return st.session_state.get('quiz_score', 0) >= 18  # 60% كحد أدنى