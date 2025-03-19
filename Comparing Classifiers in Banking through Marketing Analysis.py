#!/usr/bin/env python
# coding: utf-8

# # Bank Marketing Classification Analysis
# ### Comparing Classification Models: KNN, Logistic Regression, Decision Trees, and SVM

# ## 1. Business Understanding
# The goal of this analysis is to predict whether a client will subscribe to a term deposit based on the data from a Portuguese banking institution with a collection of the results of multiple marketing campaigns. We are comparing different classification models to determine which one performs best in predicting client behavior.

# In[1]:



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


# ## 2. Load and Explore the Dataset

# In[2]:


# Load dataset
df=pd.read_csv('/Users/anumita/Downloads/bank-additional.csv', sep=';')

# Display dataset information
print("Dataset Information:")
print(df.info())

df.head()


# ## 3. Exploratory Data Analysis (EDA)

# In[3]:



# Visualize class distribution
sns.countplot(x=df['y'])
plt.title('Class Distribution')
plt.show()

# Check correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(),annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()


# ## 4. Data Preprocessing

# In[4]:



# Binary
df['y']=df['y'].map({'yes': 1, 'no': 0})

# Label Encoding
categorical_cols = df.select_dtypes(include=['object']).columns
le =LabelEncoder()
for col in categorical_cols:
    df[col]=le.fit_transform(df[col])

# Missing values
for col in df.columns:
    if df[col].dtype =="object":  # If categorical, fill with the most frequent value
        df[col].fillna(df[col].mode()[0], inplace=True)
    else: # If numerical, fill with the median value
        df[col].fillna(df[col].median(), inplace=True)

# Split dataset
X = df.drop(columns=['y'])
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ## 5. Model Training and Evaluation

# In[10]:


import pandas as pd

# Load dataset
df = pd.read_csv("/Users/anumita/Downloads/bank-additional.csv", sep=";")

# Print
print(df.head())

# Print shape
print(f"\nDataset shape: {df.shape}")


# Check dataset 
print(f"Dataset size before cleaning: {df.shape}")

# Convert to binary (yes = 1, no = 0)
df["y"] = df["y"].map({"yes": 1, "no": 0})

# Label Encoding
categorical_cols = df.select_dtypes(include=["object"]).columns
le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Replace with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Check missing values
print("\nMissing values per column BEFORE imputation:")
print(df.isnull().sum())

# If no missing, skip imputation
if df.isnull().sum().sum() > 0:
    # Use imputation instead of dropping
    for col in df.columns:
        if df[col].dtype =="object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:  # median value
            df[col].fillna(df[col].median(), inplace=True)

# Check dataset size
print(f"\nDataset size after cleaning: {df.shape}")

# Check missing values
print("\nMissing values per column AFTER imputation:")
print(df.isnull().sum())

if df.shape[0] ==0:
    raise ValueError("Dataset is empty after cleaning. Please check the preprocessing steps.")

# Split dataset
X= df.drop(columns=["y"])
y =df["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)

# Define models
models= {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC(probability=True)
}

# Train and evaluate models 
results = {}  
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc =accuracy_score(y_test, y_pred)
    auc= roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model,'predict_proba') else 'N/A'
    results[name]= {'Accuracy': acc, 'AUC': auc}
    print(f'{name}: Accuracy = {acc:.4f}, AUC = {auc}')
    print(classification_report(y_test, y_pred))


# ## 6. Findings and Recommendations

# In[15]:


# DataFrame
results_df = pd.DataFrame(results).T

# Model Accuracy Comparison
plt.figure(figsize=(10, 5))
sns.barplot(x=results_df.index, y=results_df['Accuracy'], palette='viridis')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Classification Model')
plt.xticks(rotation=45)
plt.show()

# AUC Comparison
plt.figure(figsize=(10, 5))
sns.barplot(x=results_df.index, y=results_df['AUC'], palette='coolwarm')
plt.title('Model AUC Comparison')
plt.ylabel('AUC Score')
plt.xlabel('Classification Model')
plt.xticks(rotation=45)
plt.show()

# Insights
print("""Key Insights:
- Logistic Regression had the highest AUC (0.912), making it the best at predicting customer subscriptions.
- Support Vector Machine (SVM) performed well with an AUC of 0.906.
- Decision Trees had a lower AUC (0.674) but has better interpretability for business applications.
- KNN had a strong accuracy (0.886) but lower AUC, which may not generalize well.
""")

print("""Recommendations:
1. Use Logistic Regression for its high AUC and strong generalization ability.
2. Consider SVM since it performed well though it comes with the computational cost.
3. Use Decision Trees if interpretability is required, even though performance is slightly lower.
4. Improve results with hyperparameter tuning and Random Forest as well as Gradient Boosting.
""")


# In[ ]:




