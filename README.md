# Bank Credit Risk Prediction

This project is an end-to-end machine learning solution designed to predict the credit risk of customers from a German bank. It employs supervised learning for a binary classification task, where the target variable is labeled as 1 if a customer represents a bad credit risk and 0 if the customer represents a good risk. The model of choice for this task is a Random Forest classifier.

---

## 📁 Project Structure – **Credit Risk Prediction**

```
Credit-Risk-Prediction-German-Bank/
│
├── application.py                 # Main Flask app to serve the model as a web app
├── README.md                      # Project overview, setup instructions, and usage
├── requirements.txt               # Python dependencies for the project
│
├── artifacts/                     # Serialized model artifacts and processed datasets
│   ├── data.csv                   # Final combined dataset
│   ├── model.pkl                  # Trained Random Forest model
│   ├── preprocessor.pkl           # Preprocessing pipeline (e.g., scalers, encoders)
│   ├── train.csv                  # Training dataset
│   └── test.csv                   # Testing dataset
│
├── logs/                          # Directory for log files (runtime info, errors, etc.)
│
├── notebooks/                     # Jupyter notebooks for initial exploration and modeling
│   ├── 1. EDA.ipynb               # Exploratory Data Analysis
│   ├── 2. Modelling.ipynb         # Model building and evaluation
│   ├── data/                      # Raw dataset used during initial development
│   │   └── german_credit_data.csv
│   ├── eda_utils.py               # Helper functions for EDA
│   ├── modelling_utils.py         # Helper functions for modeling
│   └── __init__.py                # Makes notebooks a Python module (optional)
│
├── src/                           # Core source code package
│   ├── __init__.py                # Makes src a Python module
│   ├── utils.py                   # Utility functions (e.g., file handling, object saving/loading)
│   ├── exception.py               # Custom exception classes
│   ├── logger.py                  # Logger configuration
│
│   ├── components/                # Modular components for each ML step
│   │   ├── data_ingestion.py      # Loads and splits the dataset
│   │   ├── data_transformation.py # Cleans and transforms data
│   │   └── model_trainer.py       # Trains and evaluates the Random Forest model
│
│   ├── pipeline/                  # High-level training and inference pipelines
│   │   ├── train_pipeline.py      # Runs the full training pipeline end-to-end
│   │   ├── predict_pipeline.py    # Loads artifacts and makes predictions
│   └── __init__.py                # Makes pipeline a Python module
│
├── templates/                     # HTML templates for the Flask web interface
│   ├── home.html                  # Form for inputting customer data
│   └── index.html                 # Displays prediction result
│
```

---

## 🔧 Setup and Installation Instructions

1. **Create a virtual environment using Python 3.10:**

    If you don't have Python 3.10, you can try other versions such as Python 3.11 or Python 3.13, but it is recommended to use Python 3.10 for this project.
    

```bash
# Optional: Install Python 3.10 using Homebrew (macOS users only)
# Make sure Homebrew is installed first: https://brew.sh/
brew install python@3.10
```

And if you want a sentence version for context:

> **Note:** macOS users can install Python 3.10 using Homebrew with the command above. If you're using another operating system, please follow the official Python installation guidelines at [https://www.python.org/downloads/](https://www.python.org/downloads/).

   ```bash
   python3.10 -m venv .venv
   ```

2. **Activate the virtual environment:**

   * On **macOS/Linux**:

     ```bash
     source .venv/bin/activate
     ```
   * On **Windows**:

     ```bash
     .venv\Scripts\activate
     ```

3. **Install the project dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠️ Step-by-Step Guide

1. Open and Run the Notebook
    - Open the `notebooks/1. EDA.ipynb` file in VS Code.
    - Ensure the kernel is set to the `.venv` interpreter.
    - Run all cells or step through them interactively.


This EDA notebook focuses on understanding the German credit dataset to support credit risk prediction.
It begins by clearly defining the business problem and examining the structure and content of the data.
Basic statistics and distributions are analyzed to detect patterns, trends, and class imbalances.
The dataset is split into training and testing sets early to avoid data leakage.
Visualizations are generated using the custom `sns_plots()` function from `eda_utils.py`.
This function allows quick plotting of histograms, boxplots, and count plots for multiple features.
The notebook also uses `check_outliers()` from `eda_utils.py` to identify anomalies using the IQR method.
Special focus is given to relationships between customer attributes and the target variable.
The insights gained here guide data preprocessing and modeling decisions later in the pipeline.
Overall, this notebook sets the analytical foundation for building an effective credit risk model.

---

2. Open and Run the Notebook
    - Open the `notebooks/2. Modelling.ipynb` file in VS Code.
    - Ensure the kernel is set to the `.venv` interpreter.
    - Run all cells or step through them interactively.


This notebook core objective is to accurately identify customers who are likely to default—placing a strong emphasis on recall.
The process begins with reading the processed training and testing datasets.
It then proceeds to model training, where multiple algorithms are compared using cross-validation.
Evaluation is done with a focus on generalization and avoiding overfitting, using ROC-AUC as a reference.
The notebook uses helper functions (from `modelling_utils.py`) to streamline evaluation, visualization, and threshold tuning.
It includes in-depth analysis of the best model’s performance using metrics like confusion matrix and ROC curves.
Thresholds are adjusted to maximize recall while maintaining acceptable precision.
Feature importance is explored to understand which attributes most influence the model’s predictions.
By the end, a reliable, recall-optimized model is ready for production deployment.

---

3. **Run the Flask application:**

   ```bash
   python application.py
   ```

The core objective of `application.py` is to run a web-based credit risk prediction interface using Flask.
Behind the scenes, it relies on a modular backend built with well-separated components for ingestion, transformation, training, and inference.
These modules are located in the `src` directory and each plays a key role in the machine learning lifecycle.
**See the [modules descriptions below](#modules-for-applicationpy)!**

---

4. **Where will the output appear?**
Once the app is running, it will start a local server and display something like:

```
* Running on http://0.0.0.0:5001/
```

> **Note:** If port `5001` is already in use on your system, you may need to change the port number in `application.py` to another available port (for example, `5002` or `8000`). Update the line where the Flask app is run, e.g., `app.run(host="0.0.0.0", port=5002)`.

Open a **web browser** and go to:

```
http://localhost:5001/
```

or

```
http://127.0.0.1:5001/
```

From there:

   * The **home page** (`index.html`) will be shown at `/`.
   * The **form for inputting customer details** is rendered from `home.html` at `/predictdata`.
   * After submitting the form, the **prediction result** (whether the customer is a good or bad credit risk) will be displayed on the page.


---

## Modules for `application.py`

### 🔄 1. `data_ingestion.py`

This script handles loading raw data and splitting it into training and testing sets.
It ensures that the model is trained and evaluated on separate subsets to avoid overfitting.
The `TrainTestSplitConfig` class defines paths where the processed datasets will be saved.
This is typically used in the training phase (`train_pipeline.py`) but indirectly supports prediction by preparing the datasets.



### 🧼 2. `data_transformation.py`

This script is responsible for preprocessing the data before it reaches the model.
It handles encoding, scaling, and converting raw form inputs into numerical formats suitable for the model.
A transformation pipeline is saved as an artifact (`preprocessor.pkl`) and reused at prediction time.
This ensures consistency between how data was prepared during training and how it’s prepared during inference in `application.py`.



### 🤖 3. `model_trainer.py`

Here, various models are trained on the transformed data, evaluated, and the best model is saved as `model.pkl`.
It uses techniques like cross-validation and hyperparameter tuning to improve model performance.
The trained model artifact is what the Flask app loads at prediction time via `predict_pipeline.py`.


### 🔮 4. `predict_pipeline.py`

This is the most crucial script for `application.py`.
When a user submits input via the web form, `application.py` collects the data and creates an `InputData` object.
The `PredictPipeline` class loads the pre-trained model and preprocessor, transforms the input data, and generates a prediction.
This is how the web app converts form data into real-time credit risk predictions.



### 🚀 5. `train_pipeline.py`

This script brings together all components—ingestion, transformation, and model training.
It's used to automate the complete training process and generate artifacts (`model.pkl`, `preprocessor.pkl`, etc.).
Although it isn’t used at prediction time, it's essential for preparing everything that `application.py` depends on.

### Summary 

Together, these components make the web app modular, maintainable, and production-ready.
Each one abstracts a specific part of the ML lifecycle, allowing `application.py` to stay focused on handling user interaction and displaying results.


---

## Results

The deployed model accurately predicts customer credit risk for the German bank.
It classifies customers as either **good** or **bad risk** based on financial and demographic inputs.
The model achieved a **high recall**, effectively identifying most high-risk customers.
This minimizes false negatives, helping the bank reduce default-related losses.
The overall **ROC-AUC score** indicates strong discrimination between risk categories.
Precision remains acceptable, balancing risk detection with practical decision-making.
The final model generalizes well to unseen data with consistent validation results.
Key influencing factors identified include credit amount, account status, and loan duration.
A **custom threshold** was used to fine-tune sensitivity toward bad risk classification.
The model was integrated into a **Flask web application** for real-time usage.
Bank staff can input new customer data and receive instant predictions via a browser.
The system supports smarter, data-driven loan approval decisions.
It reduces reliance on manual risk evaluation and speeds up credit assessments.
The application is lightweight, accessible, and production-ready.
Model artifacts are saved and reused to maintain consistency.
The structure supports future retraining or expansion as more data becomes available.
This solution aligns with the bank’s goal of balancing profitability and risk control.
It enhances operational efficiency in the credit approval process.
The project delivers both predictive accuracy and practical utility.
Ultimately, it equips the bank with a scalable tool for responsible lending.


