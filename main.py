import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64
import re
from sklearn.preprocessing import OneHotEncoder

# ✅ Load Model
with open("car_price_model.pkl", "rb") as file:
    car_model = pickle.load(file)

# ✅ Load & Preprocess Dataset
cr = pd.read_csv('car_dheko_filled.csv')

# Clean and preprocess dataset
cr['Kms_Driven'] = cr['Kms_Driven'].fillna(0).astype(str).str.replace(',', '').astype(int)
cr['Max Power'] = cr['Max Power'].str.extract(r'(\d+\.?\d*)').astype(float)

# Get all expected features from model (if available)
if hasattr(car_model, 'feature_names_in_'):
    expected_features = list(car_model.feature_names_in_)
else:
    # Fallback to our known features if model doesn't have feature_names_in_
    expected_features = [
        'city', 'Body Type', 'Kms_Driven', 'oem', 'model', 'modelYear',
        'Fuel Type', 'Ownership', 'Transmission', 'Mileage', 'Engine Type',
        'Max Power', 'Acceleration'
    ]

# ✅ Function to Convert Local Image to Base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# ✅ Convert Background Image to Base64
bg_image_base64 = get_base64_image("machine-grey-background-volvo-wallpaper-preview.jpg")

# ✅ Apply Background Image & Custom Styling
page_bg_img = f"""
<style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg_image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    section[data-testid="stSidebar"] {{
        background: #ccd2d9 !important;
        padding: 20px;
    }}
    section[data-testid="stSidebar"] * {{
        font-family: 'Poppins', sans-serif !important;
        color: #00050a !important;
    }}
    section[data-testid="stSidebar"] h4 {{
        font-weight: 900 !important;
        text-transform: uppercase;
        text-align: center;
    }}
    section[data-testid="stSidebar"] button {{
        background: #4A00E0 !important;
        color: white !important;
        border-radius: 50px !important;
        padding: 12px !important;
        font-weight: bold !important;
        transition: 0.3s ease-in-out;
    }}
    section[data-testid="stSidebar"] button:hover {{
        background: #ccd2d9 !important;
        color: #FFF !important;
        box-shadow: 0px 0px 10px #FFFFFF88 !important;
    }}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ✅ App Title
st.title("🚗 Car Price Prediction Dashboard")

# ✅ Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Predict Price", "Data Explorer"])

# 🚗 HOME PAGE
if page == "🏠 Home":
    st.header("Welcome to the Car Price Prediction App! 🏁")
    st.write("""
    Get **AI-powered price estimates** for your car in just a few clicks!  
    Whether you're **buying, selling, or comparing**, we've got you covered.
    
    🔹 **Find out your car's worth instantly**  
    🔹 **Compare different models for better deals**  
    🔹 **Make informed decisions with data-driven insights**  
    """)

    st.markdown("---")
    st.subheader("🚀 How Does It Work?")
    st.write("""
    1️⃣ Enter your car’s **details** (Brand, Model, Year, Mileage, etc.)  
    2️⃣ Click on **Predict Price** to get an AI-driven estimate  
    3️⃣ Explore car price trends & insights 📊  
    """)

    st.markdown("---")
    st.subheader("🌟 Why Use This App?")
    st.write("""
    ✅ **Accurate Predictions** – Powered by Machine Learning  
    ✅ **Simple & Intuitive** – No complicated steps  
    ✅ **Market Insights** – Get an edge in buying & selling  
    """)

    st.markdown("---")
    st.subheader("📊 Ready to Get Started?")
    st.write("Head over to the **Prediction Page** and try it out now!")

# 📊 PREDICTION PAGE
elif page == "📊 Predict Price":
    st.header("📊 Predict Car Price")
    
    # User inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        city = st.selectbox("Select City:", options=sorted(cr['city'].unique()))
        oem = st.selectbox("Select OEM (Manufacturer):", options=sorted(cr['oem'].unique()))
        
        # Filter models based on selected OEM
        filtered_data = cr[cr['oem'] == oem]
        model_name = st.selectbox("Select Car Model:", options=sorted(filtered_data['model'].unique()))
        
        year = st.selectbox("Select the Year:", 
                          options=sorted(filtered_data['modelYear'].unique()),
                          index=len(filtered_data['modelYear'].unique())-1)
        
        kms_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)

    with col2:
        mileage = st.number_input("Mileage (km/l)", 
                                min_value=5.0, 
                                max_value=30.0, 
                                value=15.0, 
                                step=0.1)
        fuel_type = st.selectbox("Select Fuel Type:", 
                               options=sorted(filtered_data['Fuel Type'].unique()))
        transmission = st.selectbox("Select Transmission Type:", 
                                  options=sorted(filtered_data['Transmission'].unique()))
        engine_type = st.selectbox("Select Engine Type:", 
                                 options=sorted(filtered_data['Engine Type'].unique()))

    with col3:
        ownership = st.selectbox("Select Ownership Type:", 
                              options=sorted(filtered_data['Ownership'].unique()))
        max_power = st.number_input("Max Power (BHP)", 
                                  min_value=50.0, 
                                  max_value=500.0, 
                                  value=120.0, 
                                  step=5.0)
        acceleration = st.number_input("Acceleration (0-100 km/h)", 
                                     min_value=2.0, 
                                     max_value=20.0, 
                                     value=10.0, 
                                     step=0.1)
        body_type = st.selectbox("Select Body Type:", 
                              options=sorted(cr['Body Type'].unique()))

    if st.button("💰 Predict Price"):
        try:
            # Create input dictionary
            input_data = {
                'city': city,
                'oem': oem,
                'model': model_name,
                'modelYear': int(year),
                'Fuel Type': fuel_type,
                'Ownership': ownership,
                'Transmission': transmission,
                'Engine Type': engine_type,
                'Mileage': float(mileage),
                'Max Power': float(max_power),
                'Kms_Driven': int(kms_driven),
                'Acceleration': float(acceleration),
                'Body Type': body_type
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Ensure all expected features are present
            for feature in expected_features:
                if feature not in input_df.columns:
                    input_df[feature] = 0  # Add missing features with default value
            
            # Reorder columns to match model's expectations
            input_df = input_df[expected_features]
            
            # Debug information
            with st.expander("Debug Information"):
                st.write("Input DataFrame Columns:", input_df.columns.tolist())
                st.write("Model Expected Features:", expected_features)
                st.write("Input Data Sample:", input_df.head())
            
            # Make prediction
            predicted_price = car_model.predict(input_df)[0]
            
            # Display result
            st.success(f"""
            ### Predicted Price: ₹{predicted_price:,.2f}
            
            *{year} {oem} {model_name} with {kms_driven:,} km*
            """)
            
        except Exception as e:
            st.error(f"""
            🚨 Prediction Failed: {str(e)}
            
            Common issues to check:
            1. All fields are filled correctly
            2. Categorical values match exactly (case-sensitive)
            3. No special characters in text fields
            4. Numeric values are within reasonable ranges
            """)
            
            # Show sample valid values
            with st.expander("Sample Valid Values"):
                st.write("Cities:", cr['city'].unique()[:5])
                st.write("Fuel Types:", cr['Fuel Type'].unique()[:5])
                st.write("Engine Types:", cr['Engine Type'].unique()[:5])
                st.write("Body Types:", cr['Body Type'].unique()[:5])

elif page == "Data Explorer":
    st.header("🔍 Car Data Explorer")
    st.write("Explore the dataset used for predictions")
    
    # Show filtered data
    st.subheader("Filter Data")
    col1, col2 = st.columns(2)
    selected_oem = col1.selectbox("Filter by Manufacturer", options=["All"] + sorted(cr['oem'].unique().tolist()))
    selected_city = col2.selectbox("Filter by City", options=["All"] + sorted(cr['city'].unique().tolist()))
    
    filtered_data = cr.copy()
    if selected_oem != "All":
        filtered_data = filtered_data[filtered_data['oem'] == selected_oem]
    if selected_city != "All":
        filtered_data = filtered_data[filtered_data['city'] == selected_city]
    
    st.dataframe(filtered_data, height=400)
    
    # Show statistics
    st.subheader("Price Statistics")
    st.write(f"Average Price: ₹{filtered_data['price'].mean():,.2f}")
    st.write(f"Minimum Price: ₹{filtered_data['price'].min():,.2f}")
    st.write(f"Maximum Price: ₹{filtered_data['price'].max():,.2f}")
