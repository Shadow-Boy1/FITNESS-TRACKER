from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image


# Check if the user is authenticated
if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    st.warning("Please log in to access this page.")
    st.stop()
    
# Load environment variables
load_dotenv()




# Configure the Google Gemini API with a newer model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to get a response from Gemini API using the new model
def get_gemini_response(image_parts, prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')  # Switch to gemini-1.5-flash
        response = model.generate_content([image_parts[0], prompt])
        return response.text
    except Exception as e:
        return f"Error occurred: {e}"

# Function to process the uploaded image
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        return None

# Initialize Streamlit app
st.set_page_config(page_title="Calorie Tracker", layout="centered")
st.title("Calorie Tracker")

# User input and image upload
input_text = st.text_input("Input Prompt: ", key="input_text")
uploaded_file = st.file_uploader("Upload a food image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
else:
    st.warning("Please upload an image.")

# Updated predefined prompt for analysis
input_prompt = """
You are an expert nutritionist. Analyze the food items in the uploaded image and provide the following information for each item:
1. Name of the food item.
2. Calories.
3. Protein content (in grams).
4. Fat content (in grams).

Provide the analysis in this format:
1. Item Name - **Calories**, **Protein**: [x]g, **Fat**: [y]g
2. Item Name - **Calories**, **Protein**: [x]g, **Fat**: [y]g
---
**Total Calories**: [Sum]
**Total Protein**: [Sum]g
**Total Fat**: [Sum]g
"""

# Handle button click
if st.button("Analyze Food and Nutrition"):
    if uploaded_file is not None:
        image_parts = input_image_setup(uploaded_file)
        if image_parts:
            response = get_gemini_response(image_parts, input_prompt)
            st.subheader("Analysis Results")
            st.write(response)
        else:
            st.error("Failed to process the uploaded image.")
    else:
        st.error("Please upload an image before submitting.")
