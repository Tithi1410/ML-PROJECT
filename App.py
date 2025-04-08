import streamlit as st
import numpy as np
import pickle


with open('models/crop_model.pkl','rb')as model_file:
    rf = pickle.load(model_file)

with open('models/scaler (2).pkl','rb')as scaler_file:
    scaler = pickle.load(scaler_file)

reverse_crop_dict = {
    1:'rice',2:'maize',3:'chickpea',4:'kidneybeans',
    5:'pigeonpeas',6:'mothbeans',7:'mungbeans',8:'blackgram',
    9:'lentil',10:'pomegranate',11:'banana',12:'mango',
    13:'grapes',14:'watermelon',15:'muskmelon',16:'apple',
    17:'orange',18:'papaya',19:'coconut',20:'cotton',
    21:'jute',22:'coffee'
}

def recommend_crop(N,P,K,temperature,humidity,ph,rainfall):
    features = np.array([[N,K,P,temperature,humidity,ph,rainfall]])
    scaled_features = scaler.transform(features)
    prediction = rf.predict(scaled_features)[0]
    return reverse_crop_dict[prediction]

st.set_page_config(page_title="CROP RECOMMENDATION SYSTEM",layout="centered")
st.title("CROP RECOMMENDATION SYSTEM")

st.markdown("Fill the details below to get the best crop recommendation")

col1,col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen(N)",0,140,step=1)
    P = st.number_input("Phosphorous(P)",5,145,step=1)
    K = st.number_input("Potassium(K)",5,205,step=1)
    ph = st.slider("pH level",3.5,9.5,step=0.1)

with col2:
    temperature = st.slider("Temperature(C)",10.0,45.0,step=0.1)
    humidity = st.slider("Humidity(%)",10.0,100.0,step=0.1)
    rainfall = st.slider("Rainfall(mm)",0.0,300.0,step=1.0)

if st.button("Recommend Crop"):
    crop = recommend_crop(N,P,K,temperature,humidity,ph,rainfall)
    st.success(f"Recommended Crop:{crop.upper()}")
