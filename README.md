# MachineLearning-FinancialRisk-FraudDetection
 

##  Project Overview
This project applies **machine learning algorithms** on two datasets from the banking domain:  

1. **Financial Risk Assessment (Regression)**  
   - Predicting financial risk scores based on personal and financial features.  
   - Goal: Identify the most suitable regression algorithms to minimize prediction errors.  

2. **Credit Card Fraud Detection (Classification)**  
   - Classifying transactions as **fraudulent** or **legitimate**.  
   - Challenge: Severe **class imbalance** (fraud cases = ~0.17%).  

---

##  Datasets
- **Loan Dataset (Risk Assessment):** 2000 samples × 37 features  
- **Credit Card Dataset (Fraud Detection):** 284,807 transactions with 492 frauds  

---

#  Financial Risk Assessment (Regression Task)

###  1. Data Exploration
- **Samples & Features:** 2000 rows × 37 columns  
- **Missing Data:** None  
- **Duplicates:** None found  
- **Outliers:** 94 outliers in `RiskScore` (identified via IQR method)  
- **Visualization:** Boxplots and histograms used to detect anomalies  

###  2. Data Preprocessing
- **Scaling:** Applied `MinMaxScaler` to normalize quantitative features  
- **Encoding:** Used `LabelEncoder` for categorical features (`EmploymentStatus`, `EducationLevel`)  
- **Feature Selection:** Removed irrelevant columns (`ApplicationDate`, `BaseInterestRate`, `NumberOfDependents`, etc.)  
- **Final Features:** Cleaner dataset with reduced noise  

###  3. Models Implemented
- **Linear Regression** – baseline regression model  
- **Ridge Regression** – regularized linear model to reduce overfitting  
- **KNN Regressor** – tested `k=1 to 15`  
- **Neural Network (MLPRegressor)** – 4 hidden layers, max 2000 iterations → **best results**  
- **SVM Regressor** – tested with polynomial & RBF kernels (slow, higher error)  
- **Ensemble Models:**  
  - `VotingRegressor`  
  - `RandomForestRegressor`  
  - `StackingRegressor` (linear + ridge + lasso + sgd + knn)  

###  4. Model Evaluation (Regression)
#### Linear Regression
- **Train/Test Split:** 80% train, 20% test  
- **Results:**  
  - Train MSE: `3.72`, Test MSE: `4.05`  
  - Train R²: `0.94`, Test R²: `0.93`  
  - Train MAE: `1.49`, Test MAE: `1.52`

#### Ridge Regression
- **MSE (Train/Test):** `3.72 / 4.05`  
- **R² (Train/Test):** `0.938 / 0.935`  
- **MAE (Train/Test):** `1.50 / 1.53`  

#### KNN Regression
- **MSE (Test):** `62.47`  
- **R² (Test):** `-0.006` → Poor performance  
- **MAE (Test):** `6.23`  

#### Neural Network (Best Model)
- **Optimized with 4 hidden layers & logistic+ReLU activations**  
- **Lowest MSE achieved compared to other models**  

---

#  Credit Card Fraud Detection (Classification Task)

###  1. Data Exploration
- **Fraud Rate:** 492 frauds / 284,807 transactions (~0.17%)  
- **Challenge:** Class imbalance (legit >> fraud)  
- **Visualization:** Severe skewness in `Class` variable  

###  2. Data Preprocessing
- **Class Imbalance:**  
  - Applied **Random Over-Sampling** to duplicate minority fraud cases  
- **Cleaning:** Removed duplicates & missing values  
- **Scaling:** Applied `StandardScaler` for better ML performance  

###  3. Models Implemented
- **Logistic Regression** – baseline classifier  
- **KNN Classifier** – tested with `k=5`  
- **SGD Classifier** – efficient for large-scale data  
- **Neural Network (MLPClassifier)** – 5 hidden layers `(128, 64, 32, 16, 8)` with 1000 iterations  
- **Support Vector Machines (SVM)**  
  - Linear  
  - Polynomial (Poly)  
  - RBF  
- **Ensemble Models**  
  - `VotingClassifier`  
  - `RandomForestClassifier`  
  - `StackingClassifier` (Logistic + KNN + SVM as base models)  

###  4. Model Evaluation (Classification)

#### Logistic Regression (Phase 2)
- **Accuracy:** `0.9992`  
- **Precision:** `0.89`  
- **Recall:** `0.56`  
- **F1-Score:** `0.68`  

#### KNN (Phase 2)
- **Accuracy:** `0.9996`  
- **Precision:** `0.97`  
- **Recall:** `0.77`  
- **F1-Score:** `0.86`  

#### SGD
- **Accuracy:** `0.9991`  
- **Precision:** `0.88`  
- **Recall:** `0.50`  
- **F1-Score:** `0.64`  

#### SVM (RBF)
- **Accuracy:** `0.9994`  
- **Precision:** `0.98`  
- **Recall:** `0.64`  
- **F1-Score:** `0.78`  

#### Neural Network
- **Accuracy:** `0.9994`  
- **Precision:** `0.86`  
- **Recall:** `0.75`  
- **F1-Score:** `0.80`  

#### Random Forest
- **Accuracy:** `0.9995`  
- Strong balance between precision and recall  

#### Stacking Classifier (Best Model)
- **Accuracy:** `0.9996`  
- Best balance across metrics  

---

##  Model Comparison

- **Financial Risk Assessment:**  
  - Ridge Regression slightly outperformed Linear Regression.  
  - Neural Networks achieved the lowest MSE.  
  - KNN performed poorly due to dataset complexity.  

- **Fraud Detection:**  
  - Logistic Regression → strong baseline.  
  - Random Forest & Stacking Classifier → best general performance.  
  - Neural Network & SVM → captured non-linear patterns effectively.  

---

##  Conclusion
- **Financial Risk Assessment:** Ridge Regression & Neural Networks are the most suitable.  
- **Credit Card Fraud Detection:** Ensemble methods (Random Forest, Stacking) provided the **best trade-off between accuracy, precision, and recall**.  
- **Limitations:**  
  - Risk Assessment assumes linearity (nonlinear models may improve results).  
  - Fraud detection with oversampling may introduce redundancy.  

---


