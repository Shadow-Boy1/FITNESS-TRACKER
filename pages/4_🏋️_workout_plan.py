from dotenv import load_dotenv
import streamlit as st
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import google.generativeai as genai

# Load environment variables
load_dotenv()

def load_model():
    with open("workout_recommender.pkl", "rb") as file:
        model, workout_data, encoder = pickle.load(file)
    return model, workout_data, encoder

def recommend_workout(model, workout_data, user_input):
    distances, indices = model.kneighbors(user_input)
    recommendations = workout_data.iloc[indices[0]]
    return recommendations

# Load ML model
model, workout_data, encoder = load_model()

# Check if the user is authenticated
if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    st.warning("Please log in to access this page.")
    st.stop()

# Configure the Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define options
equipment_options = ["Dumbbells", "Barbell", "Kettlebell", "Resistance Bands", "Bodyweight"]
muscle_groups = ["Chest", "Back", "Legs", "Arms", "Shoulders", "Core"]
difficulty_levels = ["Beginner", "Intermediate", "Advanced"]
workout_durations = ["15 minutes", "30 minutes", "45 minutes"]

def get_gemini_response(workout):
    prompt = (
        f"Create a detailed workout plan for: {workout['Muscle Group']} \n"
        f"Using: {workout['Equipment']} \n"
        f"Difficulty: {workout['Difficulty']} \n"
        f"Duration: {workout['Duration']} \n"
        f"Workout details: {workout['Exercises']}"
    )
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([prompt])
    return response.text

# UI Components
st.title("AI-Powered Workout Plan Generator")
selected_equipment = st.multiselect("Select available equipment:", equipment_options)
selected_muscle = st.selectbox("Select target muscle group:", muscle_groups)
selected_difficulty = st.selectbox("Select difficulty level:", difficulty_levels)
selected_duration = st.selectbox("Select workout duration:", workout_durations)

if st.button("Generate Workout Plan"):
    if selected_equipment and selected_muscle and selected_difficulty and selected_duration:
        # Convert user input into a DataFrame to match training format
        user_input_df = pd.DataFrame([[selected_muscle, selected_equipment[0], selected_difficulty, selected_duration]],
                                     columns=["Muscle Group", "Equipment", "Difficulty", "Duration"])

        # Encode user input
        user_encoded = encoder.transform(user_input_df)

        # Get the best workout match
        recommended_workout = recommend_workout(model, workout_data, user_encoded)
        
        response = get_gemini_response(recommended_workout.iloc[0])
        st.markdown(response)
    else:
        st.warning("Please select all options.")
