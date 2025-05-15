# Disease Outbreak Prediction using Deep Learning

This project aims to forecast disease outbreaks using historical data, environmental variables, and demographic information. It leverages Long Short-Term Memory (LSTM) networks implemented with TensorFlow to capture temporal patterns in multi-source datasets.

## 🔍 Problem Statement
Forecasting outbreaks of diseases (e.g., dengue, flu) is vital for healthcare systems to allocate resources and implement preventive measures. Traditional models often lack adaptability to nonlinear and dynamic outbreak patterns. Deep learning offers a robust alternative by learning from diverse datasets over time.

## Data Sources
- **Epidemiological Data**: Weekly reported cases from open government health portals (simulated using synthetic datasets).
- **Weather Data**: Historical temperature, humidity, and precipitation data.
- **Demographics**: Population density, urbanization levels, and mobility metrics (proxy data).

## Technologies Used
- Python (Pandas, NumPy, Matplotlib, Seaborn)
- TensorFlow / Keras
- Scikit-learn
- Jupyter Notebook

## Model Architecture
- Data preprocessing & normalization
- Windowed time-series input generation
- LSTM layers with dropout regularization
- Dense output layer for weekly case prediction
- Evaluation metrics: MAE, RMSE, R²

## Results
- LSTM model outperformed baseline ARIMA and linear regression models
- Achieved MAE of ~7.3 cases/week on test data (simulated)
- Visualization of predicted vs actual outbreak curves showed strong alignment

## Future Work
- Incorporate mobility data from public transportation APIs
- Extend to multi-region generalization
- Deploy via Flask dashboard for real-time visualization

## Project Structure
/disease_outbreak_prediction/
├── data/
│ └── cleaned_disease_data.csv
├── notebooks/
│ └── Disease_Outbreak_Prediction.ipynb
├── models/
│ └── lstm_model.h5
├── README.md
└── requirements.txt
