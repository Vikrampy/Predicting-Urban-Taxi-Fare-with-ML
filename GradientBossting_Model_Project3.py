import streamlit as st
import pickle
import pandas as pd
import numpy as np
import math

# --- 1. Load the Trained Model ---
# This function loads the pre-trained model from the .pkl file.
@st.cache_resource
def load_model():
    try:
        with open('final_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Error: 'final_model.pkl' not found. Please ensure the model file is in the same directory.")
        return None

# --- 2. Haversine Distance Function ---
# Calculates the distance between two points on the Earth's surface.
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance_km = R * c
    # Convert kilometers to miles
    distance_miles = distance_km * 0.621371
    return distance_miles

# --- 3. Create the Streamlit UI ---
st.title("Taxi Fare Prediction App")
st.markdown("### Enter Trip Details to Predict the Fare Amount")

# Get user inputs for trip locations and time
st.header("Location and Time Details")
pickup_lat = st.number_input("Pickup Latitude", value=40.7128)
pickup_lon = st.number_input("Pickup Longitude", value=-74.0060)
dropoff_lat = st.number_input("Dropoff Latitude", value=40.7831)
dropoff_lon = st.number_input("Dropoff Longitude", value=-73.9712)

# Get user inputs for passenger count
passenger_count = st.slider("Passenger Count", min_value=1, max_value=6, value=1)

# Get user input for pickup date and time
pickup_datetime_str = st.text_input("Pickup Date and Time (YYYY-MM-DD HH:MM:SS)", "2023-10-27 15:30:00")
dropoff_datetime_str = st.text_input("Dropoff Date and Time (YYYY-MM-DD HH:MM:SS)", "2023-10-27 15:50:00")

# Create a button to trigger prediction
if st.button("Predict Fare"):
    try:
        # Convert pickup and dropoff datetime strings to datetime objects
        pickup_datetime = pd.to_datetime(pickup_datetime_str)
        dropoff_datetime = pd.to_datetime(dropoff_datetime_str)
        
        # Calculate trip duration in seconds
        trip_duration = (dropoff_datetime - pickup_datetime).total_seconds()
        
        # Calculate trip distance
        trip_distance_mile = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
        
        # Extract pickup hour
        pickup_hour = pickup_datetime.hour
        
        # Determine if it's a night trip (8 PM to 6 AM)
        is_night = 1 if (pickup_hour >= 20 or pickup_hour < 6) else 0

        # Create a DataFrame from the user inputs
        input_data = pd.DataFrame([[trip_distance_mile, pickup_hour, passenger_count, is_night]], 
                                  columns=['trip_distance_mile', 'pickup_hour', 'passenger_count', 'is_night'])
        
        # Load the model
        model = load_model()

        if model:
            # Make the prediction
            prediction = model.predict(input_data)[0]
            st.success(f"**Predicted Total Fare Amount: ${prediction:.2f}**")
            
            st.markdown("---")
            st.markdown("### Features used for prediction:")
            st.write(f"**Trip Distance:** {trip_distance_mile:.2f} miles")
            st.write(f"**Pickup Hour:** {pickup_hour}")
            st.write(f"**Is Night Trip:** {'Yes' if is_night == 1 else 'No'}")
            st.write(f"**Passenger Count:** {passenger_count}")
            st.write(f"**Trip Duration:** {trip_duration:.2f} seconds")
            
    except ValueError as ve:
        st.error(f"Error: Invalid datetime format. Please use YYYY-MM-DD HH:MM:SS. {ve}")
    except Exception as e:
        st.error(f"Prediction failed. Error: {e}")

st.markdown("""
---
**Note:** This app uses a pre-trained Gradient Boosting model.
""")
