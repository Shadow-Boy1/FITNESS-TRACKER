import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

# Sample workout data
data = {
    "Muscle Group": ["Chest", "Back", "Legs", "Arms", "Shoulders", "Core"],
    "Equipment": ["Dumbbells", "Barbell", "Kettlebell", "Bodyweight", "Resistance Bands", "Bodyweight"],
    "Difficulty": ["Beginner", "Intermediate", "Advanced", "Beginner", "Intermediate", "Advanced"],
    "Duration": ["15 minutes", "30 minutes", "45 minutes", "30 minutes", "45 minutes", "15 minutes"],
    "Exercises": [
        "Bench Press, Push-ups",
        "Pull-ups, Deadlifts",
        "Squats, Lunges",
        "Bicep Curls, Triceps Dips",
        "Shoulder Press, Lateral Raises",
        "Plank, Crunches"
    ]
}

df = pd.DataFrame(data)

# Encode categorical values using OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df[["Muscle Group", "Equipment", "Difficulty", "Duration"]])

# Train Nearest Neighbors model with better feature encoding
model = NearestNeighbors(n_neighbors=1, metric="euclidean")
model.fit(encoded_features)

# Save the trained model, dataset, and encoder
with open("workout_recommender.pkl", "wb") as file:
    pickle.dump((model, df, encoder), file)

print("✅ Model trained and saved as workout_recommender.pkl!")
