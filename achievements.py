import streamlit as st

def init_achievements():
    """تهيئة قائمة الإنجازات"""
    if 'all_achievements' not in st.session_state:
        st.session_state.all_achievements = [
            {
                'id': 'beginner',
                'title': 'المبتدئ',
                'description': 'بدء أول محاضرة',
                'icon': '🎯',
                'earned': False
            },
            {
                'id': 'first_module',
                'title': 'أول محاضرة',
                'description': 'إكمال المحاضرة الأولى',
                'icon': '📚',
                'earned': False
            },
            {
                'id': 'half_course',
                'title': 'منتصف الطريق',
                'description': 'إكمال نصف المحاضرات',
                'icon': '🏁',
                'earned': False
            },
            {
                'id': 'color_expert',
                'title': 'خبير الألوان',
                'description': 'إكمال وحدة أنظمة الألوان',
                'icon': '🎨',
                'earned': False
            },
            {
                'id': 'filter_master',
                'title': 'سيد الفلاتر',
                'description': 'إكمال وحدة الفلاتر والالتفاف',
                'icon': '🔍',
                'earned': False
            },
            {
                'id': 'edge_detector',
                'title': 'كاشف الحواف',
                'description': 'إكمال وحدة كشف الحواف',
                'icon': '📏',
                'earned': False
            },
            {
                'id': 'challenge_completer',
                'title': 'بطل التحديات',
                'description': 'إكمال جميع التحديات',
                'icon': '🏆',
                'earned': False
            },
            {
                'id': 'course_completer',
                'title': 'خبير معالجة الصور',
                'description': 'إكمال جميع المحاضرات',
                'icon': '🎓',
                'earned': False
            },
            {
                'id': 'quiz_master',
                'title': 'سيد الاختبارات',
                'description': 'الحصول على درجة كاملة في الاختبار',
                'icon': '📝',
                'earned': False
            }
        ]

def check_achievements():
    """التحقق من الإنجازات التي تم إكمالها"""
    # إنجاز المبتدئ
    if st.session_state.user_name and not get_achievement('beginner')['earned']:
        earn_achievement('beginner')
    
    # إنجاز أول محاضرة
    if st.session_state.progress.get('module1', False) and not get_achievement('first_module')['earned']:
        earn_achievement('first_module')
    
    # إنجاز منتصف الطريق
    completed = sum(st.session_state.progress.values())
    if completed >= 5 and not get_achievement('half_course')['earned']:
        earn_achievement('half_course')
    
    # إنجاز خبير الألوان
    if st.session_state.progress.get('module2', False) and not get_achievement('color_expert')['earned']:
        earn_achievement('color_expert')
    
    # إنجاز سيد الفلاتر
    if st.session_state.progress.get('module4', False) and not get_achievement('filter_master')['earned']:
        earn_achievement('filter_master')
    
    # إنجاز كاشف الحواف
    if st.session_state.progress.get('module6', False) and not get_achievement('edge_detector')['earned']:
        earn_achievement('edge_detector')
    
    # إنجاز إكمال الدورة
    if all(st.session_state.progress.values()) and not get_achievement('course_completer')['earned']:
        earn_achievement('course_completer')
    
    # إنجاز الاختبارات
    if st.session_state.quiz_score >= 90 and not get_achievement('quiz_master')['earned']:
        earn_achievement('quiz_master')

def get_achievement(achievement_id):
    """الحصول على إنجاز معين"""
    for achievement in st.session_state.all_achievements:
        if achievement['id'] == achievement_id:
            return achievement
    return None

def earn_achievement(achievement_id):
    """كسب إنجاز"""
    achievement = get_achievement(achievement_id)
    if achievement and not achievement['earned']:
        achievement['earned'] = True
        st.session_state.achievements.append({
            'title': achievement['title'],
            'description': achievement['description'],
            'icon': achievement['icon'],
            'time': st.session_state.get('start_time', '').strftime("%Y-%m-%d %H:%M:%S")
        })
        st.session_state.user_xp += 50
        st.sidebar.success(f"🎉 مبروك! لقد كسبت إنجاز: {achievement['title']}")

def display_achievements():
    """عرض الإنجازات"""
    st.sidebar.markdown("### 🏆 الإنجازات")
    
    earned = [a for a in st.session_state.all_achievements if a['earned']]
    not_earned = [a for a in st.session_state.all_achievements if not a['earned']]
    
    st.sidebar.markdown(f"**تم إكمال:** {len(earned)}/{len(st.session_state.all_achievements)}")
    
    for achievement in earned:
        st.sidebar.markdown(f"{achievement['icon']} **{achievement['title']}**")
        st.sidebar.caption(achievement['description'])
    
    if not_earned:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**الإنجازات القادمة:**")
        for achievement in not_earned:
            st.sidebar.markdown(f"🔒 {achievement['title']}")
            st.sidebar.caption(achievement['description'])

def add_achievement(title, description):
    """إضافة إنجاز للمستخدم"""
    if 'achievements' not in st.session_state:
        st.session_state.achievements = []
    
    achievement = {
        'title': title,
        'description': description,
        'time': st.session_state.get('start_time', '').strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if achievement not in st.session_state.achievements:
        st.session_state.achievements.append(achievement)
        return True
    return False
def add_achievement(title, description):
    """إضافة إنجاز للمستخدم"""
    if 'achievements' not in st.session_state:
        st.session_state.achievements = []
    
    achievement = {
        'title': title,
        'description': description,
        'time': st.session_state.get('start_time', '').strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if achievement not in st.session_state.achievements:
        st.session_state.achievements.append(achievement)
        st.session_state.user_xp = st.session_state.get('user_xp', 0) + 10
        
        # التحقق من ترقية المستوى
        if st.session_state.user_xp >= st.session_state.get('user_level', 1) * 100:
            st.session_state.user_level = st.session_state.get('user_level', 1) + 1
        
        return True
    
    return False