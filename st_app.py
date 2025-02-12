
import streamlit as st
import pandas as pd
import pickle
import os
import streamlit as st

st.set_page_config(page_title="Sales Prediction", page_icon=":bar_chart:")

st.markdown("""
    <style>
        /* Center align title */
        h1 {
            text-align: center;
            color: #1D3557;
        }
        
        /* Style the input boxes */
        .stNumberInput, .stTextInput {
            border: 2px solid #1D3557;
            border-radius: 8px;
            padding: 10px;
        }

        /* Style the sidebar */
        .stSidebar {
            background-color: #F1F4F8;
            padding: 20px;
        }
        
        /* Style the buttons */
        .stButton>button {
            background-color: #1D3557;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            padding: 10px 20px;
        }

        .stButton>button:hover {
            background-color: #457B9D;
        }

        /* Boxed layout for sections */
        .box {
            background-color: #F1F4F8;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
            margin: 20px 0px;
        }
    </style>
""", unsafe_allow_html=True)


st.write("# Sales Prediction")

product_name = st.text_input(" Type of product:", placeholder="Enter product name")

if product_name:
    st.write(f"**You entered:** {product_name}")

price = st.number_input(' Price of the product?', min_value=1, max_value=15, value=7, step=1)
ads = st.number_input(' What is the Advertisement Budget?', min_value=35, max_value=65, value=50, step=1)
promo = st.number_input(' What is the promotional budget?', min_value=35, max_value=65, value=45, step=1)


mktg_scenario = pd.DataFrame({'dollar_price': [price], 'advertisment': [ads], 'promotions': [promo]})


st.table(mktg_scenario)


if st.button("Click to predict unit sales"):
    model_path = 'lm_model_prediction.sav'
    
    if os.path.exists(model_path):  
        try:
            with open(model_path, 'rb') as file:
                loaded_model = pickle.load(file)
            
            
            pred = loaded_model.predict(mktg_scenario)[0]
            st.write(f"Predicted Unit Sales: {pred:,.0f} units")
        except Exception as e:
            st.error(f"Error loading model: {e}")
    else:
        st.error("Model file not found. Please ensure 'lm_model_prediction.sav' is in the correct directory.")


