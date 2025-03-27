import streamlit as st
import requests
import json

# Configure Google Gemini API
API_KEY = "AIzaSyAAYBVp8KJkZzlv4IkfLGd-HhURjLTfn5w"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

def generate_workout(fitness_level, goal, duration):
    prompt = f"""
    Create a structured calisthenics workout plan based on the following details:
    - **Fitness Level:** {fitness_level}
    - **Goal:** {goal}
    - **Workout Duration:** {duration} minutes
   
    Provide a detailed plan including:
    - Warm-up exercises
    - Main workout (with sets, reps, and descriptions)
    - Cool-down stretches
    - Additional tips for better performance
    """
   
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
   
    response = requests.post(API_URL, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        try:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return "Error: Unexpected response format. Please try again."
    else:
        return f"Error {response.status_code}: {response.text}"

# Streamlit UI
st.set_page_config(page_title="Calisthenics Workout Generator", layout="wide")
st.title("🏋️ Calisthenics Workout Plan Generator")
st.write("Get a customized AI-generated calisthenics workout plan tailored to your fitness level and goals.")

# User Inputs (Centered Layout)
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    fitness_level = st.selectbox("Select Your Fitness Level", ["Beginner", "Intermediate", "Advanced"])
    goal = st.selectbox("Select Your Workout Goal", ["Strength", "Endurance", "Weight Loss", "Muscle Gain"])
    duration = st.slider("Select Workout Duration (minutes)", 10, 60, 30)
    generate = st.button("Generate Workout Plan", use_container_width=True)

# Generate Workout Plan
if generate:
    with st.spinner("🔄 Generating your workout plan..."):
        workout_plan = generate_workout(fitness_level, goal, duration)
        st.subheader("📋 Your AI-Generated Workout Plan:")
        st.markdown(f"""
        <div style='background-color: #333333; color: white; padding: 15px; border-radius: 10px;'>
        {workout_plan.replace("\n", "<br>")}
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""---
🔹 *Stay consistent, track progress, and enjoy your fitness journey!*
""")

