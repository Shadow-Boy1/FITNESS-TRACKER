import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Check if the user is authenticated
if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    st.warning("Please log in to access this page.")
    st.stop()

# Load environment variables
load_dotenv()

# Configure the Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def calculate_bmi(weight, height_in_feet):
    height_in_meters = height_in_feet * 0.3048  # Convert height from feet to meters
    bmi = weight / (height_in_meters ** 2)
    return round(bmi, 2)

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 24.9:
        return "Normal weight"
    elif 25 <= bmi < 29.9:
        return "Overweight"
    else:
        return "Obese"

def get_diet_plan_gemini(bmi, bmi_category, age, gender, activity_level, dietary_preference):
    model = genai.GenerativeModel("gemini-pro")
    prompt = (f"Suggest a diet plan for a {age}-year-old {gender} with a BMI of {bmi} ({bmi_category}), "
              f"an activity level of {activity_level}, and a dietary preference of {dietary_preference}.")
    response = model.generate_content(prompt)
    return response.text if response else "Unable to fetch diet plan."

def main():
    st.title("BMI Calculator and Diet Plan Suggestion")
    
    weight = st.number_input("Enter your weight (kg)", min_value=1.0, step=0.1)
    height = st.number_input("Enter your height (feet)", min_value=1.0, step=0.1)
    age = st.number_input("Enter your age (years)", min_value=1, step=1)
    gender = st.selectbox("Select your gender", ["Male", "Female", "Other"])
    activity_level = st.selectbox("Select your activity level", ["Sedentary", "Lightly active", "Moderately active", "Very active", "Super active"])
    dietary_preference = st.selectbox("Select your dietary preference", ["Vegetarian", "Vegan", "Non-Vegetarian", "Keto", "Mediterranean", "Other"])
   
    if st.button("Calculate BMI"):
        if weight and height and age:
            bmi = calculate_bmi(weight, height)
            bmi_category = get_bmi_category(bmi)
            diet_plan = get_diet_plan_gemini(bmi, bmi_category, age, gender, activity_level, dietary_preference)
           
            st.write(f"### Your BMI: {bmi}")
            st.write(f"### Category: {bmi_category}")
            st.write(f"### Age: {age}")
            st.write(f"### Gender: {gender}")
            st.write(f"### Activity Level: {activity_level}")
            st.write(f"### Dietary Preference: {dietary_preference}")
            st.write(f"**Diet Plan:** {diet_plan}")
        else:
            st.warning("Please enter valid weight, height, and age values.")

if __name__ == "__main__":
    main()
