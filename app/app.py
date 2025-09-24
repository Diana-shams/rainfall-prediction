import streamlit as st
import pandas as pd
import joblib
import random

# Load the cleaned dataset
dataset_path = 'D:/Diana/big data/project/cleaned_dataset.csv'  # Replace with the path to your cleaned dataset
dataset = pd.read_csv(dataset_path)

# Load the trained model
model_path = "D:/Diana/big data/project/LogisticRegression_model.pkl"  # Replace with your model file path
model = joblib.load(model_path)

# Mappings for label-encoded categorical columns
location_mapping = ['Adelaide', 'Brisbane', 'Canberra', 'Darwin', 'Hobart', 'Melbourne', 'Perth', 'Sydney']
wind_gust_dir_mapping = ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW']
wind_dir_mapping = ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW']
rain_today_mapping = ['No', 'Yes']

# Set Streamlit page configuration
st.set_page_config(
    page_title="Rainfall Prediction App",
    page_icon="🌧️",
    layout="wide",  # Wide layout for better space utilization
    initial_sidebar_state="expanded",
)

# Apply custom CSS styles for a polished look
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
        }
        .stButton>button {
            background-color: #4169E1;
            color: white;
            border: none;
            padding: 0.5em 1em;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1rem;
        }
        .stButton>button:hover {
            background-color: #27408B;
        }
        .reportview-container {
            background: #FFFFFF;
            border: 1px solid #DDDDDD;
            border-radius: 10px;
            padding: 1.5em;
            margin: 2em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title for the app
st.title("🌧️ Rainfall Prediction App")
st.markdown("Predict whether it will rain tomorrow based on meteorological data.")

# Sidebar description
st.sidebar.header("About the App")
st.sidebar.write(
    """
    This app predicts the likelihood of rainfall tomorrow based on weather data. 
    You can test the model with random samples from the dataset or customize your input.
    """
)

# Initialize session state for the selected row
if "selected_row" not in st.session_state:
    st.session_state["selected_row"] = None

# Button to select a random row
st.markdown("## Test with a Random Sample")
if st.button("🎮 Select Random Row"):
    # Pick a random row
    random_row_index = random.randint(0, len(dataset) - 1)
    selected_row = dataset.iloc[random_row_index]

    # Store the selected row in session state
    st.session_state["selected_row"] = selected_row

# Check if a row is selected
if st.session_state["selected_row"] is not None:
    # Retrieve the selected row
    selected_row = st.session_state["selected_row"]

    # Display the selected row in a horizontal table
    st.markdown("### Selected Row Data:")
    st.table(pd.DataFrame(selected_row).T)

    # Pre-fill the input form with the selected row data
    st.markdown("### Input Form")
    input_data = {}

    # Divide the form into three equally distributed columns
    col1, col2, col3 = st.columns(3)

    with col1:
        input_data['Location'] = st.selectbox(
            "Location", location_mapping,
            index=int(selected_row['Location']) if int(selected_row['Location']) < len(location_mapping) else 0
        )
        input_data['MinTemp'] = st.number_input("MinTemp (°C)", value=selected_row['MinTemp'])
        input_data['MaxTemp'] = st.number_input("MaxTemp (°C)", value=selected_row['MaxTemp'])
        input_data['Rainfall'] = st.number_input("Rainfall (mm)", value=selected_row['Rainfall'])
        input_data['Evaporation'] = st.number_input("Evaporation (mm)", value=selected_row['Evaporation'])
        input_data['Sunshine'] = st.number_input("Sunshine (hours)", value=selected_row['Sunshine'])

    with col2:
        input_data['WindGustDir'] = st.selectbox(
            "WindGustDir", wind_gust_dir_mapping,
            index=int(selected_row['WindGustDir']) if int(selected_row['WindGustDir']) < len(wind_gust_dir_mapping) else 0
        )
        input_data['WindGustSpeed'] = st.number_input("WindGustSpeed (km/h)", value=selected_row['WindGustSpeed'])
        input_data['WindDir9am'] = st.selectbox(
            "WindDir9am", wind_dir_mapping,
            index=int(selected_row['WindDir9am']) if int(selected_row['WindDir9am']) < len(wind_dir_mapping) else 0
        )
        input_data['WindDir3pm'] = st.selectbox(
            "WindDir3pm", wind_dir_mapping,
            index=int(selected_row['WindDir3pm']) if int(selected_row['WindDir3pm']) < len(wind_dir_mapping) else 0
        )
        input_data['WindSpeed9am'] = st.number_input("WindSpeed9am (km/h)", value=selected_row['WindSpeed9am'])
        input_data['WindSpeed3pm'] = st.number_input("WindSpeed3pm (km/h)", value=selected_row['WindSpeed3pm'])

    with col3:
        input_data['Humidity9am'] = st.number_input("Humidity9am (%)", value=selected_row['Humidity9am'])
        input_data['Humidity3pm'] = st.number_input("Humidity3pm (%)", value=selected_row['Humidity3pm'])
        input_data['Pressure9am'] = st.number_input("Pressure9am (hPa)", value=selected_row['Pressure9am'])
        input_data['Pressure3pm'] = st.number_input("Pressure3pm (hPa)", value=selected_row['Pressure3pm'])
        input_data['Cloud9am'] = st.number_input("Cloud9am (oktas)", value=selected_row['Cloud9am'])
        input_data['Cloud3pm'] = st.number_input("Cloud3pm (oktas)", value=selected_row['Cloud3pm'])
        input_data['Temp9am'] = st.number_input("Temp9am (°C)", value=selected_row['Temp9am'])
        input_data['Temp3pm'] = st.number_input("Temp3pm (°C)", value=selected_row['Temp3pm'])
        input_data['RainToday'] = st.selectbox(
            "RainToday", rain_today_mapping,
            index=int(selected_row['RainToday']) if int(selected_row['RainToday']) < len(rain_today_mapping) else 0
        )

    # Prepare the data for prediction
    input_data_encoded = input_data.copy()
    input_data_encoded['Location'] = location_mapping.index(input_data['Location'])
    input_data_encoded['WindGustDir'] = wind_gust_dir_mapping.index(input_data['WindGustDir'])
    input_data_encoded['WindDir9am'] = wind_dir_mapping.index(input_data['WindDir9am'])
    input_data_encoded['WindDir3pm'] = wind_dir_mapping.index(input_data['WindDir3pm'])
    input_data_encoded['RainToday'] = rain_today_mapping.index(input_data['RainToday'])

    # Convert input to DataFrame for prediction
    input_df = pd.DataFrame([input_data_encoded])

    # Display the input data for prediction
    st.markdown("### Input Data for Prediction (Shape):")
    st.table(input_df)

    # Make a prediction when the user presses the button
    if st.button("Predict 🌦️"):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # Display the prediction result
        if prediction[0] == 1:
            st.success("Prediction: Yes, it will rain tomorrow.")
        else:
            st.success("Prediction: No, it will not rain tomorrow.")

       ## st.write(f"**Probability of Rain**: {prediction_proba[0][1]:.2f}")
       ## st.write(f"**Probability of No Rain**: {prediction_proba[0][0]:.2f}")
