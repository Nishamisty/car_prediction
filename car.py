import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64


def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

#set_background_image_local(r"headlights-car.jpg")


encoder_city=load_model("eencoder_city.pkl")
encoder_model=load_model("eencoder_model.pkl")
encoder_modelyear=load_model("eencoder_modelyear.pkl")
encoder_mileage=load_model("eencoder_Mileage.pkl")
encoder_fuel_type=load_model("eencoder_FuelType.pkl")
encoder_transmission=load_model("eencoder_Transmission.pkl")
encoder_ownership=load_model("eencoder_Ownership.pkl")
encoder_kms_driven=load_model("eencoder_kms_driven.pkl")
encoder_engine_type=load_model("eencoder_Engine Type.pkl")

model =load_model("car_price_model.pkl")

st.title("Car Price Prediction App")
cr = pd.read_csv('car_dheko_filled.csv')
st.write(cr)
categorical_features = ["city", "model", "modelYear", "Fuel Type", "Transmission", "Ownership","Mileage","Engine Type","Kms_Driven"]
dropdown_options = {feature: cr[feature].unique().tolist() for feature in categorical_features}

tab1, tab2 = st.tabs(["Home","Predict"])

with tab1:
    st.markdown("""
    **Welcome to the Car Price Prediction App!**
    
    This tool helps estimate car prices based on various attributes such as city, model, fuel type, year of manufacture, and more.
    
    **How it works:**
    - Enter car details in the "Predict" tab.
    - Click "Predict" to get an estimated price.
    """)

with tab2:
    a1, a2, a3 = st.columns(3)
    a4, a5, a6 = st.columns(3)
    a7, a8, a9 = st.columns(3)
    a10, a11, a12 = st.columns(3)

    with a1:
        city_select = st.selectbox("Select City", dropdown_options["city"])
        city=encoder_city.transform([[city_select]])[0][0]
    with a2:
        model_select = st.selectbox("Select Car Model", dropdown_options["model"])
        model=encoder_model.transform([[model_select]])[0][0]
    with a3:
        modelyear_select = st.selectbox("Model Type", dropdown_options["modelyear"])
        modelyear=encoder_modelyear.transform([[modelyear_select]])[0][0]
    with a4:
        fuel_type_select = st.selectbox("Fuel Type", dropdown_options["FuelType"])
        fuel_type=encoder_fuel_type.transform([[fuel_type_select]])[0][0]
    with a5:
        transmission_select = st.selectbox("Transmission Type", dropdown_options["Transmission"])
        transmission=encoder_transmission.transform([[transmission_select]])[0][0]
    with a6:
        ownership_select = st.selectbox("Ownership Count", dropdown_options["Ownership"])
        ownership=encoder_ownership.transform([[ownership_select]])[0][0]
    with a7:
        kms_driven = st.number_input("Enter KM Driven", min_value=1000, value=10000)
    with a8:
        engine_type_select = st.selectbox("Engine Type", dropdown_options["Engine Type"])
        engine_type=encoder_engine_type.transform([[engine_type_select]])[0][0]   
    with a9:
        mileage = st.number_input("Enter Mileage (kmpl)", min_value=5.0, value=15.0)

    if st.button("Predict"):
        input_data = {"city":city, "model":model,"modelyear": modelyear,"fuel_type" :fuel_type,"kms_driven" :kms_driven, 
                                "Transmission":transmission,"Engine Type":engine_type, "mileage": mileage,"Ownership": ownership}
        input_df=pd.DataFrame([input_data])
        # Call prediction function
        predicted_price = load_model.predict(input_df)
    
        
        st.subheader("Predicted Car Price")
        st.markdown(f"### :green[₹ {predicted_price[0]:,.2f}]")   