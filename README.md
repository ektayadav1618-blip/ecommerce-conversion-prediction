# E-Commerce Conversion Prediction

### Summer Analytics 2026 | Mini-Hackathon

A machine learning project focused on predicting whether an e-commerce user will convert based on demographic, browsing behaviour, traffic source, device information, and purchase history.

---

## 📌 Problem Statement

The objective of the challenge was to predict whether a user would convert (`1`) or not convert (`0`) using anonymized user-level data.

The dataset included demographic, behavioural, acquisition, device, and purchase-history features such as:

- Age
- Income
- City Tier
- Device Type
- Traffic Source
- Pages Viewed
- Products Viewed
- Time on Site
- Previous Purchase
- Discount Seen
- Browser Version
- Campaign Code

The primary evaluation metric was **F1 Score**, making the handling of class imbalance particularly important.

---

## 🔍 Exploratory Data Analysis

The training dataset contained **10,000 samples**.

The exploratory analysis revealed:

- Missing values in `Age`, `Income`, and `Time_On_Site`
- An imbalanced target variable, with fewer converted users than non-converted users
- Differences in conversion rates across traffic sources
- A positive association between discount exposure and conversion

These observations guided the preprocessing and model-selection strategy.

---

## ⚙️ Data Preprocessing

The following preprocessing pipeline was used:

- Median imputation for missing numerical values
- Standard scaling of numerical features
- One-hot encoding of categorical variables
- Stratified train-validation split

The stratified split was used to preserve the class distribution of the imbalanced target variable.

---

## 🤖 Model Development

Multiple classification models were evaluated:

| Model | F1 Score |
|---|---:|
| Logistic Regression | 0.381 |
| Random Forest | 0.384 |
| Balanced Random Forest | 0.511 |
| **Tuned Balanced Random Forest** | **0.566** |

The results showed a substantial improvement after explicitly addressing class imbalance.

---

## 🏆 Final Model

The final model was a **Tuned Balanced Random Forest** with:

- `n_estimators = 500`
- `max_depth = 10`
- `min_samples_split = 5`
- `class_weight = balanced`

### Performance

**Validation F1 Score:** 0.5659

**Public Test F1 Score:** 0.5266

The tuned Balanced Random Forest achieved the strongest F1 performance among the evaluated models and was used to generate predictions for the private test set.

---

## 💡 Key Takeaways

This project provided hands-on experience with an end-to-end supervised machine learning workflow:

**Exploratory Data Analysis → Data Preprocessing → Model Comparison → Class Imbalance Handling → Hyperparameter Tuning → Prediction Generation**

A key learning from the project was the importance of selecting modelling strategies according to both the dataset characteristics and the evaluation metric. In this case, explicitly accounting for class imbalance produced a significant improvement in F1 score.

---

## 🛠️ Technologies & Skills

- Python
- Pandas
- NumPy
- Scikit-learn
- Exploratory Data Analysis
- Data Preprocessing
- Classification
- Random Forest
- Handling Imbalanced Data
- Hyperparameter Tuning
- Model Evaluation

---

## 📁 Repository Contents

```text
ecommerce-conversion-prediction/
│
├── notebook.ipynb       # Complete machine learning workflow
├── Report.pdf           # One-page project report
└── submission.csv       # Final prediction submission
