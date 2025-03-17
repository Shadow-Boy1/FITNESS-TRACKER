import streamlit as st
from pymongo import MongoClient
import pandas as pd

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")  # Update with your MongoDB details
db = client["fitness_app"]
users_collection = db["users"]

# Function to fetch all registered users
def get_all_users():
    users = list(users_collection.find({}, {"_id": 0, "username": 1, "role": 1}))  # Fetch username & role only
    return pd.DataFrame(users) if users else None

# Function to apply styles
def apply_styles():
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #ff758c;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 5px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #ff5a85;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

# Admin Dashboard
def admin_dashboard():
    # Check if the user is authenticated and has the admin role
    if "authenticated" not in st.session_state or not st.session_state["authenticated"] or st.session_state.get("role") != "Admin":
        st.error("Unauthorized access! Redirecting to login...")
        st.session_state.clear()
        st.session_state["page"] = "Login"
        st.rerun()

    apply_styles()
    st.title("Admin Dashboard 🏋️‍♂️")

    # Fetch all registered users
    users_df = get_all_users()

    if users_df is not None:
        search_query = st.text_input("🔍 Search Users by Username:")
        
        if search_query:
            users_df = users_df[users_df["username"].str.contains(search_query, case=False, na=False)]

        st.dataframe(users_df, use_container_width=True)  # Display user data

    else:
        st.warning("No registered users found.")

    # Logout Button
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state.pop("username", None)
        st.session_state["page"] = "Login"
        st.rerun()

# Main function to handle authentication and navigation
def main():
    st.sidebar.title("Navigation")

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if "page" not in st.session_state:
        st.session_state["page"] = "Login"  # Default page

    if st.session_state["authenticated"]:
        st.session_state["page"] = "AdminDashboard"  # Redirect authenticated users

    if st.session_state["page"] == "AdminDashboard":
        admin_dashboard()
    else:
        st.error("Unauthorized access. Redirecting to login...")
        st.session_state["page"] = "Login"
        st.rerun()

if __name__ == "__main__":
    main()
