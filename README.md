#  Customer Churn Prediction (Logistic Regression)

This project predicts customer churn using a logistic regression model. The goal is to help financial institutions understand which factors contribute most to customer attrition and proactively manage churn.

---

##  Dataset
- **Source**: https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction
- **Records**: ~10,000 customers
- **Target**: `Exited` (1 = churned, 0 = retained)

---

##  Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Summary statistics
- Missing value detection
- Class distribution visualization

### 2. Data Preprocessing
- Dropped irrelevant identifiers (`CustomerId`, `Surname`)
- Used `SimpleImputer` for mean imputation
- Scaled numeric features using `StandardScaler`
- Converted categorical features via one-hot encoding

### 3. Modeling
- **Algorithm**: Logistic Regression  
- **Pipeline**: Imputer → Scaler → Model  
- **Hyperparameter**: `max_iter=2000`

### 4. Evaluation
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- Feature Importance plot

---

##  Results

- **Accuracy**: `0.8091`
- **Class 0 (Not Churned)**:
  - Precision: 0.83 | Recall: 0.96 | F1-score: 0.89
- **Class 1 (Churned)**:
  - Precision: 0.57 | Recall: 0.21 | F1-score: 0.31

<p align="center">
  <img src="images/confusion_matrix.png" width="400"/>
</p>

Despite good overall accuracy, the model shows imbalance in recall for churned customers. This highlights the need for future enhancement via SMOTE or ensemble models.

---

##  Top Features Influencing Churn

<p align="center">
  <img src="images/feature_importance.png" width="400"/>
</p>

Key drivers:
- **Age**
- **Inactive Membership**
- **Geography: Germany**
- **Gender: Male**

---

##  Tools Used
- Python: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- Notebook: `Jupyter`
- Modeling: Logistic Regression (Pipeline)
- Visualization: Confusion matrix, feature importance bar chart

---

##  How to Run This Project

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
jupyter notebook notebooks/churn_analysis.ipynb
