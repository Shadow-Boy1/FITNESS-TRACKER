import streamlit as st
import requests
import pandas as pd
import os
from datetime import datetime, timedelta
import random
from dotenv import load_dotenv
from functools import lru_cache


# Load environment variables from .env
load_dotenv()

# Load API key from .env or secrets.toml
SPOONACULAR_API_KEY = os.getenv('SPOONACULAR_API_KEY') or st.secrets.get('SPOONACULAR_API_KEY')

if not SPOONACULAR_API_KEY:
    st.error("API key not found! Please set it in the .env file or secrets.toml.")

# ======================== #
#      HELPER FUNCTIONS    #
# ======================== #
@st.cache_data(ttl=timedelta(hours=24))
def calculate_calories(weight, height, age, gender, activity_level, goal):
    """Calculates daily calorie needs using Harris-Benedict equation"""
    if gender == "Male":
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    
    activity_multiplier = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725
    }[activity_level]
    
    tdee = bmr * activity_multiplier
    
    if goal == "Weight Loss":
        return tdee * 0.8
    elif goal == "Muscle Gain":
        return tdee * 1.1
    else:
        return tdee

@st.cache_data(ttl=timedelta(hours=12))
def get_spoonacular_recipes(params):
    """Fetch recipes from Spoonacular API with caching"""
    base_url = "https://api.spoonacular.com/recipes/complexSearch"
    params['apiKey'] = SPOONACULAR_API_KEY
    params['addRecipeNutrition'] = True
    params['fillIngredients'] = True
    params['addRecipeInformation'] = True  # Get full recipe details
    params['number'] = 5  # Reduced to save API calls
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get('results', [])
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 402:
            st.error("API limit reached - using cached recipes")
            return []
        st.error(f"Error fetching recipes: {e}")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return []

def process_recipe(recipe):
    """Enhanced recipe processing with more details"""
    nutrition = recipe.get('nutrition', {})
    nutrients = {item['name']: item['amount'] for item in nutrition.get('nutrients', [])}
    
    recipe_data = {
        'name': recipe['title'],
        'calories': nutrients.get('Calories', 0),
        'carbs': nutrients.get('Carbohydrates', 0),
        'protein': nutrients.get('Protein', 0),
        'fat': nutrients.get('Fat', 0),
        'fiber': nutrients.get('Fiber', 0),
        'diabetes_friendly': (nutrients.get('Carbohydrates', 0) <= 45 and 
                            nutrients.get('Sugar', 0) <= 10),
        'glycemic_index': 'Low' if nutrients.get('Carbohydrates', 0) <= 30 else 'Medium',
        'ingredients': [ing['name'] for ing in recipe.get('extendedIngredients', [])],
        'image': recipe.get('image', ''),
        'id': recipe['id'],
        'prep_time': recipe.get('readyInMinutes', 'N/A'),
        'recipe_url': f"https://spoonacular.com/recipes/{recipe['title'].replace(' ', '-')}-{recipe['id']}",
        'instructions': []
    }
    
    # Add cooking instructions if available
    if 'analyzedInstructions' in recipe and recipe['analyzedInstructions']:
        recipe_data['instructions'] = [
            step['step'] for step in recipe['analyzedInstructions'][0]['steps']
        ]
    
    return recipe_data

def generate_meal_plan(duration, diabetes_status, target_carbs, low_gi_only, dietary_pref, calories):
    """Generate multi-day meal plan using Spoonacular API"""
    meal_plans = {}
    
    for day in range(duration):
        meal_plan = {}
        total_carbs = 0
        
        meal_types = {
            "Breakfast": {'mealType': 'breakfast', 'maxCarbs': target_carbs},
            "Lunch": {'mealType': 'lunch', 'maxCarbs': target_carbs},
            "Dinner": {'mealType': 'dinner', 'maxCarbs': target_carbs},
            "Snacks": {'mealType': 'snack', 'maxCarbs': target_carbs//2}
        }
        
        for meal_type, params in meal_types.items():
            query_params = {
                'type': params['mealType'],
                'maxCarbs': params['maxCarbs'],
                'maxCalories': calories//4 if meal_type != "Snacks" else calories//8
            }
            
            # Apply dietary restrictions
            if "Vegetarian" in dietary_pref:
                query_params['diet'] = 'vegetarian'
            if "Vegan" in dietary_pref:
                query_params['diet'] = 'vegan'
            if "Gluten-Free" in dietary_pref:
                query_params['intolerances'] = 'gluten'
            if "Keto" in dietary_pref:
                query_params['diet'] = 'keto'
            
            # Diabetes-specific filters
            if diabetes_status != "None" and low_gi_only:
                query_params['maxCarbs'] = params['maxCarbs'] * 0.8
            
            recipes = get_spoonacular_recipes(query_params)
            processed_recipes = [process_recipe(r) for r in recipes]
            
            if diabetes_status != "None":
                processed_recipes = [
                    r for r in processed_recipes 
                    if r['diabetes_friendly'] and 
                    (not low_gi_only or r['glycemic_index'] == 'Low')
                ]
            
            if processed_recipes:
                selected = random.choice(processed_recipes)
                meal_plan[meal_type] = selected
                total_carbs += selected['carbs']
        
        meal_plans[f"Day {day+1}"] = {'meals': meal_plan, 'total_carbs': total_carbs}
    
    return meal_plans

# ======================== #
#    STREAMLIT UI LAYOUT   #
# ======================== #

st.title("🍏 Diabetes-Friendly Meal Prep Planner")
st.markdown("### Personalized nutrition planning with Spoonacular API")

# Initialize session state
if 'user_prefs' not in st.session_state:
    st.session_state.user_prefs = None

# Sidebar for User Inputs
with st.sidebar:
    st.header("👤 User Profile")
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    gender = st.radio("Gender", ["Male", "Female", "Other"])
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    activity_level = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])
    
    st.header("🩺 Health Conditions")
    diabetes_status = st.selectbox("Diabetes Status", ["None", "Type 1", "Type 2", "Prediabetes", "Gestational"])
    if diabetes_status != "None":
        target_carbs = st.slider("Max carbs per meal (g)", 20, 100, 45)
        low_gi_only = st.checkbox("Prioritize Low-Glycemic Index (GI) foods?")
    else:
        target_carbs = 100
        low_gi_only = False
    
    st.header("🎯 Goals & Preferences")
    goal = st.selectbox("Health Goal", ["Weight Loss", "Maintenance", "Muscle Gain"])
    dietary_pref = st.multiselect("Dietary Preferences", ["Vegetarian", "Vegan", "Keto", "Gluten-Free", "Low-Sodium"])
    plan_duration = st.selectbox("Plan Duration", [1, 3, 7], format_func=lambda x: f"{x} day{'s' if x>1 else ''}")
    
    if st.button("💾 Save Preferences"):
        st.session_state.user_prefs = {
            'age': age,
            'gender': gender,
            'dietary_pref': dietary_pref,
            'goal': goal,
            'diabetes_status': diabetes_status,
            'target_carbs': target_carbs,
            'low_gi_only': low_gi_only
        }
        st.success("Preferences saved!")
    
    if st.session_state.user_prefs and st.button("🗑️ Clear Saved Preferences"):
        st.session_state.user_prefs = None
        st.experimental_rerun()

# Load saved preferences
if st.session_state.user_prefs:
    st.sidebar.info("ℹ️ Using saved preferences")
    if st.sidebar.button("↩️ Restore Saved Preferences"):
        prefs = st.session_state.user_prefs
        age = prefs['age']
        gender = prefs['gender']
        dietary_pref = prefs['dietary_pref']
        goal = prefs['goal']
        diabetes_status = prefs['diabetes_status']
        target_carbs = prefs['target_carbs']
        low_gi_only = prefs['low_gi_only']
        st.experimental_rerun()

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    # Calculate and display calorie target
    calories = calculate_calories(weight, height, age, gender, activity_level, goal)
    st.success(f"📊 **Daily Calorie Target:** {int(calories)} kcal")
    
    # Generate and display meal plan
    if st.button("🍽️ Generate Meal Plan", use_container_width=True):
        with st.spinner(f'Generating {plan_duration}-day meal plan...'):
            meal_plans = generate_meal_plan(
                plan_duration,
                diabetes_status, 
                target_carbs, 
                low_gi_only, 
                dietary_pref, 
                calories
            )
        
        if not any(day['meals'] for day in meal_plans.values()):
            st.error("No recipes found matching your criteria. Try relaxing some filters.")
        else:
            for day, plan in meal_plans.items():
                with st.expander(f"### {day} Meal Plan (Total Carbs: {int(plan['total_carbs'])}g)", expanded=True):
                    for meal, details in plan['meals'].items():
                        st.markdown(f"**{meal}**: {details['name']}")
                        col_img, col_nut = st.columns([1, 2])
                        
                        with col_img:
                            if details['image']:
                                st.image(details['image'], width=150)
                        
                        with col_nut:
                            st.write(f"🔹 Calories: {int(details['calories'])} kcal")
                            st.write(f"🔹 Carbs: {int(details['carbs'])}g")
                            st.write(f"🔹 Protein: {int(details['protein'])}g")
                            st.write(f"🔹 Fiber: {int(details['fiber'])}g")
                            if details['prep_time'] != 'N/A':
                                st.write(f"🔹 Prep Time: {details['prep_time']} min")
                            if diabetes_status != "None":
                                st.write(f"🔹 Glycemic Index: {details['glycemic_index']}")
                        
                        # Instructions Section
                        if st.button("📝 View Instructions", key=f"instr_{day}_{meal}"):
                            if details['instructions']:
                                st.write("**Cooking Instructions:**")
                                for i, step in enumerate(details['instructions'], 1):
                                    st.write(f"{i}. {step}")
                            else:
                                st.markdown(f"[View complete recipe on Spoonacular]({details['recipe_url']})")
                        
                        # Ingredients Section
                        if st.checkbox("Show ingredients", key=f"ing_{day}_{meal}"):
                            st.write("**Ingredients:**")
                            for ing in details['ingredients']:
                                st.write(f"- {ing}")
            
            # Diabetes-Specific Tips
            if diabetes_status != "None":
                st.info("""
                💡 **Diabetes Management Tips**:
                - Pair carbs with protein/fiber to reduce glucose spikes
                - Space meals evenly throughout the day
                - Monitor your blood sugar responses to different foods
                """)

with col2:
    # Grocery List Generator
    if 'meal_plans' in locals() and meal_plans:
        st.subheader("🛒 Multi-Day Grocery List")
        ingredients = {}
        
        for day, plan in meal_plans.items():
            for meal, details in plan['meals'].items():
                for ing in details['ingredients']:
                    ingredients[ing] = ingredients.get(ing, 0) + 1
        
        grocery_list = st.container()
        for item, count in sorted(ingredients.items()):
            if count > 1:
                grocery_list.checkbox(f"{item} ({count}x)", key=f"grocer_{item}")
            else:
                grocery_list.checkbox(item, key=f"grocer_{item}")
        
        if st.button("📥 Export Grocery List"):
            grocery_text = "=== Grocery List ===\n"
            grocery_text += "\n".join(
                f"{item} ({count}x)" if count > 1 else item 
                for item, count in sorted(ingredients.items())
            )
            st.download_button(
                label="Download List",
                data=grocery_text,
                file_name="grocery_list.txt",
                mime="text/plain"
            )
    
    # Nutrition Tips Section
    st.subheader("💡 Smart Nutrition Tips")
    
    with st.expander("View Personalized Tips"):
        if diabetes_status != "None":
            st.write("- Choose whole grains over refined carbs")
            st.write("- Include healthy fats (avocados, nuts) in each meal")
            st.write("- Monitor portion sizes of carb-heavy foods")
        
        if "Weight Loss" in goal:
            st.write("- Focus on protein-rich foods to stay full longer")
            st.write("- Drink water before meals to reduce calorie intake")
        
        if "Vegan" in dietary_pref:
            st.write("- Combine plant proteins (beans + rice) for complete amino acids")
            st.write("- Consider B12 and iron supplementation")
        
        if "Keto" in dietary_pref:
            st.write("- Maintain net carbs below 20g for ketosis")
            st.write("- Increase healthy fat intake for energy")
    
    # API Status
    st.markdown("---")
    st.caption(f"🔌 Powered by Spoonacular API | Credits remaining: {st.secrets.get('API_CREDITS', 'Unlimited')}")

# ======================== #
#       HOW TO RUN         #
# ======================== #
# 1. Create a .env file with SPOONACULAR_API_KEY=your_api_key
# 2. Install requirements: pip install requests python-dotenv
# 3. Run with: streamlit run meal_prep_app.py
