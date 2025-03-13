import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64
import re

# ✅ Load Model
with open("car_price_model.pkl", "rb") as file:
    car_model = pickle.load(file)

# ✅ Load & Preprocess Dataset
cr = pd.read_csv('car_dheko_filled.csv')

# Clean and preprocess dataset
cr['Kms_Driven'] = cr['Kms_Driven'].fillna(0).astype(str).str.replace(',', '').astype(int)
cr['Max Power'] = cr['Max Power'].str.extract(r'(\d+\.?\d*)').astype(float)

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
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Predict Price", "ℹ️ Car Details"])

# 🚗 HOME PAGE
# --------------------------------
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
    
    # 🚀 How it Works
    st.subheader("🚀 How Does It Work?")
    st.write("""
    1️⃣ Enter your car’s **details** (Brand, Model, Year, Mileage, etc.)  
    2️⃣ Click on **Predict Price** to get an AI-driven estimate  
    3️⃣ Explore car price trends & insights 📊  
    """)

    st.markdown("---")
    
    # 🌟 Why Use This App?
    st.subheader("🌟 Why Use This App?")
    st.write("""
    ✅ **Accurate Predictions** – Powered by Machine Learning  
    ✅ **Simple & Intuitive** – No complicated steps  
    ✅ **Market Insights** – Get an edge in buying & selling  
    """)

    st.markdown("---")

    st.subheader("📊 Ready to Get Started?")
    st.write("Head over to the **Prediction Page** and try it out now!")

# --------------------------------
 # Define the required features
    feature_columns = [
    'city', 'Body Type', 'Kms_Driven', 'oem', 'model', 'modelYear',
    'Fuel Type', 'Ownership', 'Transmission', 'Mileage', 'Engine Type',
    'Max Power', 'Acceleration']
# 📊 PREDICTION PAGE
elif page == "📊 Predict Price":
    st.header("📊 Predict Car Price")
    
    # User inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        city = st.selectbox("Select City:", options=cr['city'].unique())
        oem = st.selectbox("Select OEM (Manufacturer):", options=cr['oem'].unique())
        
        # Filter models based on selected OEM
        filtered_data = cr[cr['oem'] == oem]
        model_name = st.selectbox("Select Car Model:", options=filtered_data['model'].unique())
        
        # Ensure the year is within a reasonable range
        year = st.selectbox("Select the Year:", options=sorted(filtered_data['modelYear'].unique()), index=0)
        
        # Default value for kilometers driven
        kms_driven = st.number_input("Kilometers Driven", min_value=0, value=10000)

    with col2:
        Mileage = st.selectbox("Select Mileage (in km/l):", options=sorted(filtered_data['Mileage'].unique()))
        fuel_type = st.selectbox("Select Fuel Type:", options=filtered_data['Fuel Type'].unique())
        transmission = st.selectbox("Select Transmission Type:", options=filtered_data['Transmission'].unique())
        engine_type = st.selectbox("Select Engine Type:", options=sorted(filtered_data['Engine Type'].unique()))

    with col3:
        ownership = st.selectbox("Select Ownership Type:", options=filtered_data['Ownership'].unique())
        max_power = st.selectbox("Select Max Power (in BHP):", options=sorted(filtered_data['Max Power'].unique()))
        Acceleration = st.selectbox("Select Acceleration Type:", options=sorted(filtered_data['Acceleration'].unique()))
        body_type = st.selectbox("Select Body Type:", options=sorted(cr['Body Type'].unique()))  # Add body type selection
        # Prediction Page
    # Prediction Page
    if st.button("💰 Predict Price"):
        try:
            # Encode categorical variables
            city_encoded = cr['city'].astype('category').cat.categories.get_loc(city)
            oem_encoded = cr['oem'].astype('category').cat.categories.get_loc(oem)
            model_name_encoded = cr['model'].astype('category').cat.categories.get_loc(model_name)
            year_encoded = year  # Assuming year is already a numerical value
            fuel_type_encoded = cr['Fuel Type'].astype('category').cat.categories.get_loc(fuel_type)
            ownership_encoded = cr['Ownership'].astype('category').cat.categories.get_loc(ownership)
            transmission_encoded = cr['Transmission'].astype('category').cat.categories.get_loc(transmission)
            engine_type_encoded = cr['Engine Type'].astype('category').cat.categories.get_loc(engine_type)
            body_type_encoded = cr['Body Type'].astype('category').cat.categories.get_loc(body_type)

            # Prepare input data
            input_data = np.array([[city_encoded, oem_encoded, model_name_encoded, year_encoded, 
                                    fuel_type_encoded, ownership_encoded, transmission_encoded, 
                                    engine_type_encoded, Mileage, max_power, kms_driven, 
                                    Acceleration, body_type_encoded]])  # Ensure all features are included

            # Check the shape of input data
            #st.write("Input data shape:", input_data.shape)  # Should output (1, 23)

            # Debugging: Print encoded values
            #st.write("Encoded values:", input_data)

            # Predict price
            predicted_price = car_model.predict(input_data)[0]
            st.success(f"Predicted Price: ₹{predicted_price:,.2f}")
        except Exception as e:
            st.error("An error occurred while predicting the price. Please check your inputs and try again.")
            st.write(f"Error details: {str(e)}")


# ✅ Car Details Page
elif page == "ℹ️ Car Details":
    st.header("ℹ️ Car Details")
    car_search = st.text_input("Search for a Car Model")
    if st.button("🔍 Search"):
        filtered_data = cr[cr['model'].str.contains(car_search, case=False, na=False)]
        st.write(filtered_data.head() if not filtered_data.empty
                  else "No matching cars found.")
