# Air-Quality-Index-Forecasting-via-GA-KELM

Forecast tomorrow’s air quality—today. This project leverages a hybrid Genetic Algorithm (GA) and Kernel Extreme Learning Machine (KELM) model to accurately forecast Air Quality Index (AQI). The model is trained on real-world air quality datasets to predict AQI levels. GA is optimal feature selection and hyperparameter and KELM provides fast &amp; robust nonlinear regression performance.

# 🚀 What is GA-KELM?

GA (Genetic Algorithm): Mimics natural selection to find the best features and fine-tune model parameters.
KELM (Kernel Extreme Learning Machine): A lightning-fast, single-layer neural network using kernel tricks to capture complex patterns.
Together, they create a hybrid model that predicts air quality more efficiently than traditional methods.

# 🔍 Why This Project?

Air pollution affects health, productivity, and quality of life. Real-time and accurate AQI forecasting helps:

1.🏙️ Smart cities plan better

2.🏥 Hospitals prepare for respiratory risks

3.🌿 Citizens make informed daily choices

# 📦 Features

✅ GA-based feature selection

✅ Nonlinear regression via KELM

✅ Train/test on real AQI datasets

✅ Visual insights with graphs

✅ Clean, modular codebase

# 📁 Project Structure

📂 GA-KELM-AQI

├── data/                # Air quality datasets (CSV format)

├── ga_kelm_model.py     # GA + KELM hybrid implementation

├── utils.py             # Helper functions for preprocessing & metrics

├── visualize.py         # Plotting results

├── main.py              # Entry point to train and predict

└── README.md            # You're here!

# ⚙️ How to Run

Step 1: Clone the repo
git clone https://github.com/yourusername/GA-KELM-AQI.git
cd GA-KELM-AQI

Step 2: Install required packages
pip install -r requirements.txt

Step 3: Run the model
python main.py

# 📊 Sample Output
You'll get:

Forecasted AQI vs actual values

Error metrics like RMSE, MAE

Feature importance via GA

Beautiful visual plots!

# 📚 Dataset
You can use any real-world AQI dataset. Common sources:

UCI Machine Learning Repository

OpenAQ

Government air quality portals

# 🤖 Future Ideas
Integrate real-time AQI APIs

Add LSTM or hybrid models

Deploy as a web service or mobile app

# 📬 Contact
Made with ❤️ by G.Venkatesh

📧 gvenkatesh9193@example.com
