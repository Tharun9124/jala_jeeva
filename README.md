# Marine AI Dashboard - README

## Project Overview  
The JALA JEEVA is an AI-powered application that detects plastic pollution in ocean images using a pre-trained MobileNetV2 model. It provides real-time predictions through image uploads and webcam capture, tracks prediction history, and visualizes data using interactive graphs and maps.

## Features  
- **Home Page:** Introduction to the app with usage tips.  
- **Prediction Page:** Upload images for AI-based classification (Clean Ocean or Polluted Ocean).  
- **Webcam Page:** Capture images from your webcam for real-time predictions.  
- **Graphs with Map:** Data visualization using Altair charts and interactive maps with Folium.  
- **History Page:** View a log of past predictions stored in a CSV file.  

## Technologies Used  
- **Python**  
- **Streamlit:** Frontend framework for building the web interface.  
- **TensorFlow/Keras:** For deep learning and image classification using MobileNetV2.  
- **OpenCV:** Image preprocessing and manipulation.  
- **Altair:** Data visualization library for generating charts.  
- **Folium:** For rendering interactive maps.  
- **streamlit-webrtc:** For real-time webcam capture and processing.

## Installation  

1. **Clone the repository:**  
   ```bash  
   git clone <repository_url>  
   cd <repository_directory>  
   ```  

2. **Install the required packages:**  
   ```bash  
   pip install streamlit tensorflow opencv-python-headless altair folium streamlit-folium streamlit-webrtc  
   ```  

3. **Run the Streamlit app:**  
   ```bash  
   streamlit run app.py  
   ```  

## File Structure  
```
├── app.py                   # Main application script  
├── marine_ai_model.h5       # Saved Keras model  
├── predictions.csv          # CSV file storing prediction history  
└── README.md                # Project documentation  
```  

## Usage  

1. **Home Page:**  
   Provides an overview and usage tips.

2. **Prediction:**  
   - Upload an image (JPEG/PNG) from your device.  
   - The model predicts whether the image shows a "Clean Ocean" or "Polluted Ocean".  
   - Save the prediction to the history log if desired.

3. **Webcam:**  
   - Capture images using your webcam directly in the app.  
   - Real-time predictions are displayed with confidence scores.

4. **Graphs with Map:**  
   - Displays sample graphs and an interactive map to enhance visualization.  

5. **History:**
   - Shows all previous predictions stored in `predictions.csv`.  

## Customization  

### Background Image  
You can set a custom background image by modifying the `set_background()` function with a new image URL.  

### Model Customization  
To use a different model, update the `create_model()` function and replace the `marine_ai_model.h5` file.

## Troubleshooting  

- **Model Not Found Error:** If `marine_ai_model.h5` is missing, the app will automatically create a new one.  
- **Prediction History Missing:** If `predictions.csv` is not found, it will be created after the first prediction is saved.  

## Contributions  
Contributions are welcome! Feel free to fork the repository and submit pull requests.  

## Contact  
For questions or support, please contact: yashaswiniyash2004@gmail.com
