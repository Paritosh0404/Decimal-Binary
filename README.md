# 🌦️ Envision 2025: Weather Prediction System

**Team Name**: Decimal-Binary  
**Team Code**: EN43  
**Competition**: Envision Datathon 2025

## 📋 Project Overview

The Weather Prediction System is developed to forecast weather conditions using machine learning models trained on historical data. The project encompasses data collection, preprocessing, visualization, model training, and deployment through a web interface.

## 👥 Team Members

- Paritosh
- Ayush
- Mayur
- Elasha

## 🗂️ Repository Contents

- `Total_data.csv`: Raw weather data from [Visual Crossing](https://www.visualcrossing.com/).
- `preprocessed_weather_data.csv`: Cleaned and preprocessed dataset used for model training.
- `weather_data_visualization.pbix`: Power BI dashboard for data visualization and insights.
- `app.py`: Flask application for deploying the trained weather prediction model.
- `index.html`: Frontend interface for user interaction with the prediction system.
- `decimal_binary_document.pdf`: Detailed project documentation, including methodologies and findings.
- `round_3_doc.pdf`: Solution plan and approach for the third round of the competition.
- `Team.pdf`: Information about the team and project summary.

## 🔍 Data Collection and Preprocessing

- **Source**: Weather data was scraped from [Visual Crossing](https://www.visualcrossing.com/).
- **Preprocessing**: The raw data underwent cleaning and transformation to ensure quality inputs for the predictive model. Detailed steps are documented in `decimal_binary_document.pdf`.

## 📊 Data Visualization

Utilized Power BI to create an interactive dashboard (`weather_data_visualization.pbix`) that provides insights into weather patterns and trends, aiding in feature selection and model evaluation.

## 🤖 Model Training and Deployment

- **Model**: Trained a machine learning model on the preprocessed dataset to predict weather conditions.
- **Deployment**: Developed a Flask-based web application (`app.py`) with an HTML frontend (`index.html`) to allow users to input data and receive weather predictions.

## 🚀 How to Run the Web Application

1. **Setup**:
   - Ensure all required `.pkl` files (model artifacts) are in the project directory.
   - Place `index.html` inside a folder named `templates`.

2. **Run the Application**:
   - Execute the Flask app:
     ```bash
     python app.py
     ```
   - Open the provided URL (e.g., `http://127.0.0.1:5000/`) in a web browser.

3. **Usage**:
   - Enter the required input data in the provided fields.
   - Click on the "Predict" button to view the weather prediction.

## 📝 Documentation

For a comprehensive understanding of the project, including methodologies, data preprocessing steps, model details, and solution approaches, refer to the following documents:

- `decimal_binary_document.pdf`
- `round_3_doc.pdf`
