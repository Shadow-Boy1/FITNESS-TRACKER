import streamlit as st
from pymongo import MongoClient
import bcrypt

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")  # Update as needed
db = client["fitness_app"]
users_collection = db["users"]

# Function to hash password
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# Function to verify password
def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# Function to add a new user or admin
def add_user(username, password, role):
    if users_collection.find_one({"username": username}):
        return False  # User already exists
    users_collection.insert_one({"username": username, "password": hash_password(password), "role": role})
    return True

# Function to validate user credentials and return role
def validate_user(username, password):
    user = users_collection.find_one({"username": username})
    if user and verify_password(password, user["password"]):
        return user["role"]  # Return the user's role
    return None

# Styled UI
def apply_styles():
    st.markdown("""
        <style>
        .stTextInput, .stButton {
            margin-top: 15px;
        }
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

# Login Page
def login_page():
    apply_styles()
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if not username or not password:
            st.error("Please enter both username and password")
        else:
            role = validate_user(username, password)
            if role:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.session_state["role"] = role
                st.session_state["page"] = "Admin Dashboard" if role == "Admin" else "User Dashboard"
                st.rerun()
            else:
                st.error("Invalid username or password")

    if st.button("Signup"):
        st.session_state["page"] = "Signup"
        st.rerun()

# Signup Page
def signup_page():
    apply_styles()
    st.title("Signup Page")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    role = st.radio("Select Role", ["User", "Admin"])

    if st.button("Signup"):
        if not username or not password or not confirm_password:
            st.error("Please fill in all fields")
        elif password != confirm_password:
            st.error("Passwords do not match")
        elif add_user(username, password, role):
            st.success("Account created successfully! Redirecting to Login...")
            st.session_state["page"] = "Login"
            st.rerun()
        else:
            st.error("Username already exists. Choose another.")

# User Dashboard
def user_dashboard():
    st.title("User Dashboard")
    st.write(f"Welcome, {st.session_state['username']}!")
    st.write("Here you can access your workout features.")

    if st.button("Logout"):
        st.session_state.clear()
        st.session_state["page"] = "Login"
        st.rerun()

# Admin Dashboard - Now Fully Protected
def admin_dashboard():
    # Redirect if not an admin
    if "role" not in st.session_state or st.session_state["role"] != "Admin":
        st.error("Unauthorized access! Redirecting to login...")
        st.session_state.clear()
        st.session_state["page"] = "Login"
        st.rerun()

    st.title("Admin Dashboard")
    st.write(f"Welcome, {st.session_state['username']}! Here you can view insights about users.")

    # Fetch and display all users
    users = list(users_collection.find({}, {"_id": 0, "username": 1, "role": 1}))
    
    if users:
        st.write("### Registered Users:")
        for user in users:
            st.write(f"👤 **Username:** {user['username']} | 🏷 **Role:** {user['role']}")
    else:
        st.warning("No registered users found.")

    if st.button("Logout"):
        st.session_state.clear()
        st.session_state["page"] = "Login"
        st.rerun()

# Main function
def main():
    st.sidebar.title("Navigation")
    
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    
    if "page" not in st.session_state:
        st.session_state["page"] = "Login"

    # Enforce Role-Based Navigation
    if st.session_state["authenticated"]:
        if st.session_state["role"] == "Admin":
            st.session_state["page"] = "Admin Dashboard"
        elif st.session_state["role"] == "User":
            if st.session_state["page"] == "Admin Dashboard":  # Prevent manual admin access
                st.error("Unauthorized access! Redirecting to User Dashboard...")
                st.session_state["page"] = "User Dashboard"
                st.rerun()
        else:
            st.session_state["page"] = "Login"
            st.rerun()

    if st.session_state["page"] == "Login":
        login_page()
    elif st.session_state["page"] == "Signup":
        signup_page()
    elif st.session_state["page"] == "User Dashboard":
        user_dashboard()
    elif st.session_state["page"] == "Admin Dashboard":
        admin_dashboard()
    else:
        st.session_state["page"] = "Login"
        st.rerun()

if __name__ == "__main__":
    main()
