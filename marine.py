import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
import numpy as np
import cv2
import pandas as pd
import altair as alt
import folium
from folium.plugins import HeatMap  # Added for heatmap
from streamlit_folium import st_folium
from PIL import Image
import io
import base64

# Tips to display
TIPS = [
    "Tip 1: Use the webcam to get real-time predictions.",
    "Tip 2: Upload clear images for better prediction accuracy.",
    "Tip 3: Use the 'Graphs with Map' section for data visualization."
]

# Set the background image for all pages
def set_background():
    background_style = f"""
    <style>
    .stApp {{
        background-image: url("https://rare-gallery.com/uploads/posts/1140797-digital-art-simple-background-water-minimalism-fish-blue-waves-gradient-underwater-bubbles-biology-drop-wave-computer-wallpaper-macro-photography-marine-biology-deep-se.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)


# Function to display rolling tips
def display_tips():
    tips_style = """
    <style>
    .rolling-tips {
        font-size: 16px;
        color: #FFFFFF;
        padding: 10px;
        margin-bottom: 15px;
        animation: scroll-tips 10s linear infinite;
        white-space: nowrap;
        overflow: hidden;
        display: block;
        text-align: center;
        border-radius: 5px;
    }

    @keyframes scroll-tips {
        from {
            transform: translateX(100%);
        }
        to {
            transform: translateX(-100%);
        }
    }
    </style>
    <div class="rolling-tips">
        """ + " | ".join(TIPS) + """
    </div>
    """
    st.markdown(tips_style, unsafe_allow_html=True)



# Sidebar Navigation using Dropdown
st.sidebar.title("Marine AI Dashboard")
navigation = st.sidebar.selectbox("Navigate", ["Home", "Prediction", "Webcam", "Graphs with Map", "History", "Business Idea", "Marine Ecosystem VR/AR"])

# Function to create the AI model
def create_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load or create the model
def load_or_create_model():
    try:
        model = load_model('marine_ai_model.h5')
        print("Loaded existing model.")
    except:
        model = create_model()
        model.save('marine_ai_model.h5')
        print("New model created.")
    return model

model = load_or_create_model()

# Preprocess image for prediction
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Predict the uploaded image
def predict_image(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    classes = ['Clean Ocean', 'Polluted Ocean']
    return classes[np.argmax(predictions)], np.max(predictions)

# Save predictions to CSV
def save_to_csv(data):
    df = pd.DataFrame(data, columns=["Image_Name", "Prediction", "Confidence"])
    df.to_csv("predictions.csv", index=False, mode='a', header=False)

# Display predictions history from CSV
def display_prediction_history():
    try:
        df = pd.read_csv("predictions.csv", names=["Image_Name", "Prediction", "Confidence"])
        st.dataframe(df)
    except FileNotFoundError:
        st.write("No predictions saved yet.")

# Function to handle webcam stream
def capture_webcam_image():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()  # To update the webcam feed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display webcam feed in Streamlit
        stframe.image(frame, channels="BGR", use_column_width=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to capture image
            cap.release()
            cv2.destroyAllWindows()
            return frame  # Return the captured frame

    cap.release()
    cv2.destroyAllWindows()
    return None

# Function to handle webcam stream prediction
def predict_webcam_image():
    st.title("Capture an Image from Webcam")

    # Capture webcam image
    captured_image = capture_webcam_image()
    
    if captured_image is not None:
        # Display the captured image
        st.image(captured_image, caption="Captured Image", use_column_width=True)
        
        # Get the prediction for the captured image
        prediction, confidence = predict_image(captured_image)
        
        # Display the prediction
        st.write(f"Prediction: {prediction}")
        st.write(f"Confidence: {confidence * 100:.2f}%")
        
        # Optionally save the prediction
        if st.button("Save Prediction"):
            save_to_csv([["Captured Image", prediction, f"{confidence * 100:.2f}%"]])
            st.success("Prediction saved to predictions.csv")


# Business Idea Page
def display_business_idea():
    st.title("Business Idea: Marine Ecosystem Conservation")
    st.write(""" 
        **Marine AI - Empowering Marine Conservation Through Technology**

        Our mission is to leverage AI and computer vision to monitor and protect marine ecosystems. 
        The Marine AI system uses a deep learning model to detect plastic pollution in ocean images, 
        helping researchers, NGOs, and government bodies take proactive actions to protect the oceans.

        **Business Opportunity**:
        - **Target Market**: Environmental organizations, government agencies, and corporations involved in sustainability and marine protection.
        - **Revenue Model**: Subscription-based API for pollution monitoring, custom consulting services for data analysis, and reporting tools.
        - **Impact**: Our AI-powered platform enables real-time pollution detection, data-driven decision-making, and helps achieve SDG 14 (Life Below Water).
        
        We are currently looking for partners and investors who are passionate about sustainability and willing to help scale this solution globally.
    """)
    # New Section for Company Credentials and Pricing
    st.subheader("Partner with Us - Provide Your Credentials and Pricing for Ocean Cleaning")
    
    # Form for companies to enter their information
    with st.form(key='company_form'):
        company_name = st.text_input("Company Name")
        contact_name = st.text_input("Contact Person Name")
        contact_email = st.text_input("Contact Email")
        phone_number = st.text_input("Phone Number")
        cleaning_price = st.number_input("Price for Ocean Cleaning (per km²)", min_value=0.0, format="%.2f")
        recyclable_price = st.number_input("Price for Taking Recyclable Items (per kg)", min_value=0.0, format="%.2f")
        
        # Submit Button
        submit_button = st.form_submit_button("Submit")

        # When the form is submitted, save the data
        if submit_button:
            if company_name and contact_name and contact_email and phone_number and cleaning_price and recyclable_price:
                # Create a dictionary of the data
                company_data = {
                    "Company Name": company_name,
                    "Contact Person": contact_name,
                    "Contact Email": contact_email,
                    "Phone Number": phone_number,
                    "Ocean Cleaning Price (per km²)": cleaning_price,
                    "Recyclable Items Price (per kg)": recyclable_price
                }
                
                # Convert to DataFrame
                df = pd.DataFrame([company_data])
                
                # Save the data to a CSV file (or a database)
                try:
                    df_existing = pd.read_csv("company_partners.csv")
                    df_existing = pd.concat([df_existing, df], ignore_index=True)
                    df_existing.to_csv("company_partners.csv", index=False)
                except FileNotFoundError:
                    df.to_csv("company_partners.csv", index=False)
                
                st.success("Your credentials and pricing have been submitted successfully.")
                st.write("Thank you for partnering with us! We will get in touch with you soon.")
            else:
                st.error("Please fill in all the fields before submitting.")

# Marine Ecosystem VR/AR Page (New Page)
def display_vr_ar():
    st.title("Marine Ecosystem VR/AR Experience")
    st.write("""
        Immerse yourself in the beauty of marine ecosystems and explore the effects of pollution in virtual reality (VR) or augmented reality (AR).
        
        **How to Explore**:
        - **Step 1**: Click the button below to experience the virtual tour of the ocean and coral reefs.
        - **Step 2**: Learn about the impact of pollution and conservation efforts in real-time.
    """)
    
    # Link to WebXR or VR Experience
    st.subheader("Explore the Ocean!")
    html_code = """
    <iframe src="https://thehydro.us/xr-experiences" width="100%" height="600" frameborder="0"></iframe>
    """
    st.markdown(html_code, unsafe_allow_html=True)



# Main Page Content Rendering
if navigation == "Home":
    set_background()  # Set background image for Home page
    display_tips()  # Display tips above header
    st.title("JALA JEEVA")
    st.subheader("Marine Ecosystem Monitoring")
    st.write(""" This app uses AI to detect plastic pollution in ocean images. It leverages a pre-trained MobileNetV2 model to classify images as either "Clean Ocean" or "Polluted Ocean." """)

elif navigation == "Prediction":
    set_background()  # Set background image for other pages
    display_tips()  # Display tips above header
    st.title("Upload an Image for Prediction")
    uploaded_file = st.file_uploader("Upload an Ocean Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image = np.array(image)
        
        # Get prediction
        prediction, confidence = predict_image(image)
        st.write(f"Prediction: {prediction}")
        st.write(f"Confidence: {confidence * 100:.2f}%")
        
        # Option to save the prediction
        if st.button("Save Prediction"):
            save_to_csv([[uploaded_file.name, prediction, f"{confidence * 100:.2f}%"]])
            st.success("Prediction saved to predictions.csv")

elif navigation == "Webcam":
    set_background()  # Set background image for other pages
    display_tips()  # Display tips above header
    predict_webcam_image()

elif navigation == "Graphs with Map":
    set_background()  # Set background image for other pages
    display_tips()  # Display tips above header
    st.title("Analysis with Graphs and Maps")
    
    # Example mock data for pollution density (latitude, longitude, intensity)
    pollution_data = [
        [37.7749, -122.4194, 0.6],  # San Francisco, intensity 0.6
        [34.0522, -118.2437, 0.8],  # Los Angeles, intensity 0.8
        [40.7128, -74.0060, 0.5],   # New York, intensity 0.5
        [51.5074, -0.1278, 0.7],    # London, intensity 0.7
        [35.6762, 139.6503, 0.9]    # Tokyo, intensity 0.9
    ]
    
    # Convert pollution data into a DataFrame
    df_pollution = pd.DataFrame(pollution_data, columns=["Latitude", "Longitude", "Intensity"])
    
    # Create the base map
    map_center = [20.0, 0.0]  # Global center
    zoom_start = 2  # Zoom level for global view
    map = folium.Map(location=map_center, zoom_start=zoom_start)
    
    # Create a heatmap using the pollution data
    heat_data = [[row["Latitude"], row["Longitude"], row["Intensity"]] for _, row in df_pollution.iterrows()]
    HeatMap(heat_data).add_to(map)
    
    # Render the map in Streamlit
    st.write("This heatmap shows pollution density in various ocean regions.")
    st_folium(map, width=700, height=500)

elif navigation == "History":
    set_background()  # Set background image for other pages
    display_tips()  # Display tips above header
    display_prediction_history()

elif navigation == "Business Idea":
    set_background()  # Set background image for other pages
    display_tips()  # Display tips above header
    display_business_idea()

elif navigation == "Marine Ecosystem VR/AR":
    set_background()  # Set background image for other pages
    display_tips()  # Display tips above header
    display_vr_ar()
