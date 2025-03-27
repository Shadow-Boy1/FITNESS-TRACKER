import streamlit as st
import requests
import json

# Configure Google Gemini API
API_KEY = "AIzaSyCFRsm9e4_CNVl3T3Mao4JTAZ9xs4ZesxA"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

def get_fitness_advice(user_query):
    prompt = f"""
    You are a professional fitness trainer and nutritionist. Answer the user's question based on scientific research and best practices.
   
    **User Query:** {user_query}
   
    Provide a concise, helpful, and motivational response.
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
st.set_page_config(page_title="AI Fitness Chatbot", layout="wide")
st.title("💬 AI Fitness Chatbot")
st.write("Ask any fitness or nutrition-related question, and get expert AI-powered advice!")

# User Input
user_query = st.text_input("Enter your fitness or diet question:", placeholder="e.g., What’s the best way to increase push-up strength?")
ask_button = st.button("Ask AI", use_container_width=True)

# Generate Response
if ask_button and user_query:
    with st.spinner("🔄 Fetching expert advice..."):
        advice = get_fitness_advice(user_query)
        st.subheader("🤖 AI Response:")
        st.markdown(f"""
        <div style='background-color: #333333; color: white; padding: 15px; border-radius: 10px;'>
        {advice.replace("\n", "<br>")}
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""---
💡 *Stay healthy and keep pushing towards your fitness goals!*""")
