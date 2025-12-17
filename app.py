import streamlit as st
import pandas as pd
import json
import os
from curriculum import CURRICULUM
from utils import init_gemini, get_gemini_response, evaluate_submission

# --- CONFIG ---
st.set_page_config(page_title="AION Trainer", page_icon="🧠", layout="wide")

# --- THEME CONFIG ---
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

if "daily_progress" not in st.session_state:
    st.session_state.daily_progress = {}

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am AION, your AI Engineering Coach. Ready to train?"}]

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712009.png", width=60)
    st.title("AION Coach")
    st.caption("AI Engineering Curriculum")
    
    # Theme Toggle
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    
    st.markdown("---")

# Define Theme Colors based on toggle
if st.session_state.dark_mode:
    # Dark Slate Theme
    bg_color = "#0F172A"
    sidebar_bg = "#1E293B"
    card_bg = "#1E293B"
    text_color = "#F8FAFC"
    subtext_color = "#94A3B8"
    border_color = "#334155"
else:
    # Light Clean Theme
    bg_color = "#FFFFFF"
    sidebar_bg = "#F3F4F6"
    card_bg = "#FFFFFF"
    text_color = "#111827"
    subtext_color = "#6B7280"
    border_color = "#E5E7EB"

# Inject Dynamic CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Overrides */
    [data-testid="stAppViewContainer"] {{
        background-color: {bg_color};
    }}
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg};
        border-right: 1px solid {border_color};
    }}
    
    h1, h2, h3, p, label, .stMarkdown, .stRadio label {{
        font-family: 'Inter', sans-serif;
        color: {text_color} !important;
    }}
    
    p, .stMarkdown p {{
        color: {text_color} !important;
    }}
    
    /* Stat Cards */
    .stat-card {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}
    .stat-card h4 {{
        color: {subtext_color} !important;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
    }}
    .stat-card h2 {{
        color: {text_color} !important;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }}
    
    /* Hero Card */
    .hero-card {{
        background: linear-gradient(135deg, {sidebar_bg} 0%, {bg_color} 100%);
        border: 1px solid {border_color};
        padding: 32px;
        border-radius: 16px;
        margin-bottom: 32px;
    }}
    
    /* Nav Items (Custom) */
    .nav-row {{
        display: flex; 
        align-items: center; 
        padding: 12px; 
        background: {card_bg}; 
        border-radius: 8px; 
        margin-bottom: 8px; 
        border: 1px solid {border_color};
        transition: transform 0.1s;
    }}
    .nav-row:hover {{
        transform: translateX(4px);
        border-color: #3B82F6;
    }}
    
    /* Expander Styling */
    [data-testid="stExpander"] details > summary {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px;
    }}
    
    [data-testid="stExpander"] details {{
        border: none !important;
    }}
    
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {{
        color: {text_color} !important;
    }}
    
    /* Buttons explicitly styled */
    .stButton button {{
        background: linear-gradient(90deg, #2563EB 0%, #4F46E5 100%);
        color: white !important;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
        transition: transform 0.1s;
    }}
    .stButton button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
    }}
    
</style>
""", unsafe_allow_html=True)

# Navigation
with st.sidebar:
    mode = st.radio("Navigate", ["Dashboard", "Daily Trainer", "Coach Chat", "Portfolio", "Settings"], label_visibility="collapsed")
    
    st.markdown("---")
    
    # Progress
    completed_days = len([d for d, p in st.session_state.daily_progress.items() if p.get("submitted")])
    progress_percent = int((completed_days / 168) * 100)
    st.write(f"**Progress**: {progress_percent}%")
    st.progress(progress_percent / 100)
    
    st.markdown("---")
    # Settings inline
    # Prioritize st.secrets for Cloud deployment
    api_key = None
    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        # Secrets not found (local dev), fallback to env
        pass

    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        with st.expander("🔐 API Key Needed"):
            val = st.text_input("Gemini Key", type="password")
            if val:
                os.environ["GEMINI_API_KEY"] = val
                init_gemini(val)
                st.rerun()
    else:
        # Ensure configured if found from secrets/env
        init_gemini(api_key)
        st.success("✅ Connected")

# --- PAGES ---

def get_day_data(day_num):
    return next((d for d in CURRICULUM if d["day"] == day_num), None)

if mode == "Dashboard":
    
    # Hero Section
    st.markdown("""
    <div class="hero-card">
        <h1>Welcome back, Engineer.</h1>
        <p>You are on the path to mastery. Consistency is your algorithms, practice is your data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <h4>Curriculum Day</h4>
            <h2>1 / 168</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <h4>Current Streak</h4>
            <h2>1 Day</h2>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <h4>Projects Shipped</h4>
            <h2>0</h2>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <h4>XP Earned</h4>
            <h2>0 XP</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🗺️ Your Journey")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Phases
    # Group by phase with chronological smoothing
    phases = {}
    current_real_phase = "Phase 1: Foundations" # Fallback
    
    sorted_curriculum = sorted(CURRICULUM, key=lambda x: x['day'])
    
    for day in sorted_curriculum:
        p = day.get("phase", "Unknown")
        
        # If generic phase name, treat as belonging to current real phase context
        if p in ["Integration", "Review", "Unknown"]:
            target_phase = current_real_phase
        else:
            current_real_phase = p
            target_phase = p
            
        if target_phase not in phases: 
            phases[target_phase] = []
        phases[target_phase].append(day)
        
    for phase_name, days in phases.items():
        with st.expander(f"{phase_name} ({len(days)} days)", expanded=True):
            for d in days[:5]: 
                is_done = st.session_state.daily_progress.get(d["day"], {}).get("submitted")
                icon = "✅" if is_done else "⚪"
                
                # Custom Row with interpolated styling
                st.markdown(f"""
                <div class="nav-row">
                    <span style="font-size: 1.2rem; margin-right: 16px;">{icon}</span>
                    <div style="flex-grow: 1;">
                        <strong style="color: {text_color};">Day {d['day']}: {d['title']}</strong><br>
                        <span style="font-size: 0.85rem; color: {subtext_color};">{d['time']} • {d['goal']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            if len(days) > 5:
                st.caption(f"and {len(days)-5} more...")

elif mode == "Daily Trainer":
    st.header("Daily Training Module 🥋")
    
    # Day Selector
    selected_day = st.number_input("Select Day", min_value=1, max_value=168, value=1)
    day_data = get_day_data(selected_day)
    
    if not day_data:
        st.error("Day not found!")
    else:
        st.subheader(f"Day {selected_day}: {day_data['title']}")
        st.caption(f"{day_data['phase']} | Time: {day_data['time']}")
        
        # Tabs
        tab_learn, tab_build, tab_quiz, tab_submit = st.tabs(["📚 Learn", "🔨 Build", "📝 Quiz", "✅ Submit"])
        
        # --- KEY: Cache generated content in session state to save API calls ---
        doc_key = f"day_{selected_day}_content"
        if doc_key not in st.session_state:
            st.session_state[doc_key] = {"lesson": None, "guide": None, "quiz": None}

        with tab_learn:
            st.info(f"**Goal**: {day_data['goal']}")
            
            # Dynamic Lesson Generation
            if st.session_state[doc_key]["lesson"]:
                st.markdown(st.session_state[doc_key]["lesson"])
                if st.button("🔄 Regenerate Lesson"):
                    st.session_state[doc_key]["lesson"] = None
                    st.rerun()
            else:
                st.markdown("### 🧠 AI Lesson Generator")
                st.write("Click below to have AION Coach generate a personalized mini-lesson and learning path for today's concepts.")
                if st.button("Generate Lesson"):
                    if not os.getenv("GEMINI_API_KEY"):
                        st.error("Please set API Key in Settings first.")
                    else:
                        with st.spinner("Coach is preparing your lesson..."):
                            prompt = f"""
                            Create a detailed, interactive lesson for Day {selected_day}: "{day_data['title']}".
                            Goal: {day_data['goal']}
                            Concepts: {', '.join(day_data['concepts'])}
                            
                            Structure:
                            1. **Introduction**: exciting hook about why this matters.
                            2. **Deep Dive**: Explain the concepts simply but technically. Use analogies.
                            3. **How to Navigate**: Suggest a path for the learner (e.g., "Read this docs first, then try X").
                            4. **Resources**: Suggest 2-3 types of resources to search for (e.g. "Search for 'ReAct pattern python'").
                            
                            Format: Markdown.
                            """
                            lesson = get_gemini_response(prompt, system_instruction="You are an expert technical instructor.")
                            st.session_state[doc_key]["lesson"] = lesson
                            st.rerun()

            st.markdown("---")
            st.markdown("### Coach Prompts (Manual)")
            for p in day_data.get("prompts", []):
                st.code(p, language="text")

        with tab_build:
            st.markdown("### 🛠️ Build Guide")
            
            # Dynamic Build Guide
            if st.session_state[doc_key]["guide"]:
                st.markdown(st.session_state[doc_key]["guide"])
                if st.button("🔄 Regenerate Guide"):
                    st.session_state[doc_key]["guide"] = None
                    st.rerun()
            else:
                st.info("Need help? Generate a step-by-step guide with code snippets.")
                if st.button("Generate Step-by-Step Guide"):
                    if not os.getenv("GEMINI_API_KEY"):
                        st.error("Please set API Key in Settings first.")
                    else:
                        with st.spinner("Architecting solution..."):
                            prompt = f"""
                            Create a step-by-step implementation guide for Day {selected_day}.
                            Build Tasks: {day_data['build']}
                            Tech Stack: Python, Streamlit, Gemini.
                            
                            Output:
                            - Detailed steps with code blocks (Python/Bash).
                            - Explain *why* we are doing each step.
                            - Show file structure context if needed.
                            """
                            guide = get_gemini_response(prompt, system_instruction="You are a Senior Staff Engineer guiding a junior.")
                            st.session_state[doc_key]["guide"] = guide
                            st.rerun()
            
            st.divider()
            st.subheader("Checklist")
            # Checklist
            current_progress = st.session_state.daily_progress.get(selected_day, {})
            checked_steps = current_progress.get("checked_steps", [])
            
            new_checked = []
            for i, step in enumerate(day_data['build']):
                is_checked = st.checkbox(step, value=(i in checked_steps), key=f"step_{selected_day}_{i}")
                if is_checked:
                    new_checked.append(i)
            
            # Save progress
            if new_checked != checked_steps:
               if selected_day not in st.session_state.daily_progress:
                   st.session_state.daily_progress[selected_day] = {}
               st.session_state.daily_progress[selected_day]["checked_steps"] = new_checked

            st.markdown("### Streamlit Integration Task")
            st.info(day_data.get('streamlit_integration', 'No UI task today.'))

        # --- KEY: Cache generated content in session state to save API calls ---
        doc_key = f"day_{selected_day}_content"
        if doc_key not in st.session_state:
            st.session_state[doc_key] = {"lesson": None, "guide": None, "quiz": None}

        # ... (Learn and Build tabs omitted for brevity, logic remains same but doc_key now has quiz) ...

        with tab_quiz:
            st.markdown("### 📝 Knowledge Check")
            
            if st.session_state[doc_key]["quiz"]:
                quiz_data = st.session_state[doc_key]["quiz"]
                try:
                    quiz_json = json.loads(quiz_data)
                    
                    # Use 'with' to ensure text and widgets are unified in the form
                    with st.form(key=f"quiz_form_{selected_day}"):
                        user_answers = {}
                        for idx, q in enumerate(quiz_json):
                            st.markdown(f"**Q{idx+1}: {q['question']}**")
                            
                            if q['type'] == 'multiple_choice':
                                user_answers[idx] = st.radio("Select Answer:", q['options'], key=f"q_{selected_day}_{idx}", label_visibility="collapsed")
                            else:
                                user_answers[idx] = st.text_input("Your Answer:", key=f"q_{selected_day}_{idx}", label_visibility="collapsed")
                            
                            st.divider()
                        
                        submitted = st.form_submit_button("Submit Quiz")
                        
                        if submitted:
                            score = 0
                            for idx, q in enumerate(quiz_json):
                                ua = user_answers.get(idx)
                                correct = q['answer']
                                # Simple normalization for checking
                                if str(ua).strip().lower() == str(correct).strip().lower():
                                    score += 1
                                else:
                                    st.warning(f"Q{idx+1} Incorrect. Answer: {correct}")
                                    
                            st.success(f"You scored {score}/{len(quiz_json)}!")
                            if score == len(quiz_json):
                                st.balloons()
                            
                            # Update progress
                            if selected_day not in st.session_state.daily_progress:
                                st.session_state.daily_progress[selected_day] = {}
                            # Only overwrite strict score if higher? Simple logic for now.
                            st.session_state.daily_progress[selected_day]["quiz_score"] = score

                    if st.button("🔄 Regenerate Quiz"):
                        st.session_state[doc_key]["quiz"] = None
                        st.rerun()

                except json.JSONDecodeError:
                    st.error("Error parsing quiz. Please regenerate.")
                    st.code(quiz_data)
                    if st.button("Retry Generation"):
                        st.session_state[doc_key]["quiz"] = None
                        st.rerun()
            else:
                st.info("Ready to test your knowledge? Generate a custom quiz for today's topics.")
                if st.button("Generate Quiz (5-10 Questions)"):
                    if not os.getenv("GEMINI_API_KEY"):
                        st.error("Please set API Key in Settings first.")
                    else:
                        with st.spinner("Coach is writing a quiz..."):
                            prompt = f"""
                            Generate a challenging technical quiz for: "{day_data['title']}".
                            Concepts: {', '.join(day_data['concepts'])}
                            Difficulty: Hard.
                            
                            Requirements:
                            - 5 to 7 Questions.
                            - Mix of 'multiple_choice' and 'short_answer'.
                            - Return ONLY valid JSON array. No markdown formatting.
                            
                            JSON Schema:
                            [
                              {{
                                "question": "string",
                                "type": "multiple_choice",
                                "options": ["A", "B", "C", "D"],
                                "answer": "Exact Text of Correct Option"
                              }},
                              {{
                                "question": "string",
                                "type": "short_answer",
                                "answer": "Expected Answer Key"
                              }}
                            ]
                            """
                            # Use JSON mode if possible, or just strict instructions
                            quiz_res = get_gemini_response(prompt, system_instruction="You are a strict exam creator. Output JSON only.")
                            # Strip potential markdown code blocks
                            clean_json = quiz_res.replace("```json", "").replace("```", "").strip()
                            st.session_state[doc_key]["quiz"] = clean_json
                            st.rerun()

        with tab_submit:
            st.markdown("### Proof of Work")
            
            st.write("**Deliverables Required:**")
            for d in day_data['deliverables']:
                st.markdown(f"- {d}")
                
            submission_text = st.text_area("Paste code snippet or repo link here:", height=200)
            
            if st.button("Submit & Evaluate"):
                if not os.getenv("GEMINI_API_KEY"):
                    st.error("Please set API Key in Settings first.")
                else:
                    with st.spinner("Coach is reviewing..."):
                        feedback = evaluate_submission(submission_text, context=str(day_data))
                        st.markdown("### Coach Feedback")
                        st.markdown(feedback)
                        
                        # Mark complete
                        if selected_day not in st.session_state.daily_progress:
                             st.session_state.daily_progress[selected_day] = {}
                        st.session_state.daily_progress[selected_day]["submitted"] = True
                        st.success("Day marked as Complete!")

elif mode == "Coach Chat":
    st.header("Coach AION 🤖")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("Ask your coach..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            if not os.getenv("GEMINI_API_KEY"):
                response = "Please set your API Key in settings."
            else:
                response = get_gemini_response(prompt, system_instruction="You are AION Coach, a tough but encouraging AI Engineering mentor.")
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

elif mode == "Portfolio":
    st.header("Your Portfolio 📂")
    st.write("Projects required for graduation:")
    
    projects = [
        {"name": "Classical ML Capstone", "deadline": "Day 48", "status": "Pending"},
        {"name": "Deep Learning Vision App", "deadline": "Day 75", "status": "Pending"},
        {"name": "NLP Capstone", "deadline": "Day 98", "status": "Pending"},
        {"name": "RAG + Agent App", "deadline": "Day 140", "status": "Pending"},
        {"name": "Final Production Capstone", "deadline": "Day 160", "status": "Pending"}
    ]
    
    df = pd.DataFrame(projects)
    st.table(df)

elif mode == "Settings":
    st.header("Settings")
    st.write("Configure your learning environment.")
    
    st.write("Current API Key Status: " + ("✅ Set" if os.getenv("GEMINI_API_KEY") else "❌ Not Set"))
    
    if st.button("Reset Progress"):
        st.session_state.daily_progress = {}
        st.experimental_rerun()
