#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import pickle
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error



# יבוא הפונקציה prepare_data מהקובץ car_data_prep.py
from car_data_prep import prepare_data

# שלב 1: קריאת הנתונים
# נניח שהקובץ dataset.csv נמצא באותה תיקייה
df = pd.read_csv('dataset.csv')

# שלב 2: עיבוד הנתונים באמצעות הפונקציה prepare_data()
df = prepare_data(df)


# In[ ]:


# קידוד משתנים קטגוריאליים באמצעות One-Hot Encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# בחירת המשתנים החשובים (Features) והמשתנה התלוי (Target)
X = df_encoded.drop('Price', axis=1)  # כל העמודות מלבד מחיר (Price)
y = df_encoded['Price']  # העמודה של המחיר היא ה-Target

# פיצול הנתונים ל-Train ו-Test ביחס של 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# הצגת צורת הנתונים לאחר הפיצול
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


# יצירת עותקים של X_train ו-X_test כדי להימנע מ-SettingWithCopyWarning
X_train = X_train.copy()
X_test = X_test.copy()

# הוספת התכונה החדשה ל-X_train ול-X_test
X_train['Km_per_Year'] = df.loc[X_train.index, 'Km_per_Year']
X_test['Km_per_Year'] = df.loc[X_test.index, 'Km_per_Year']


# In[ ]:


# יצירת אובייקט אימפוטר שמשלים ערכים חסרים עם ממוצע העמודה
imputer = SimpleImputer(strategy='mean')

# השלמת ערכים חסרים בסט האימון והבדיקה
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# יצירת מודל רנדום פורסט לבחירת התכונות החשובות ביותר
forest = RandomForestRegressor(random_state=42)
forest.fit(X_train_imputed, y_train)

# קבלת חשיבות התכונות מהמודל
importances = forest.feature_importances_

# סידור התכונות לפי החשיבות
indices = np.argsort(importances)[::-1]

# בחירת מספר מסוים של תכונות חשובות (למשל, 20)
num_features = 12
selected_features = indices[:num_features]

# הצגת התכונות החשובות ביותר
plt.figure(figsize=(10, 6))
plt.title("Feature importances")
plt.bar(range(num_features), importances[selected_features], align="center")
plt.xticks(range(num_features), [X_train.columns[i] for i in selected_features], rotation=90)
plt.show()

# הפחתת התכונות בסטים
X_train_reduced = X_train_imputed[:, selected_features]
X_test_reduced = X_test_imputed[:, selected_features]

# הצגת צורת הנתונים החדשה
X_train_reduced.shape, X_test_reduced.shape


# In[ ]:


model = ElasticNetCV(alphas=np.logspace(-4, 0, 50), l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95], cv=10, random_state=42)
model.fit(X_train_reduced, y_train)

# חיזוי על סט הבדיקה
y_pred = model.predict(X_test_reduced)

# חישוב RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# הצגת RMSE והמשקלים
rmse, sorted(zip(model.coef_, [X_train.columns[i] for i in selected_features]), reverse=True, key=lambda x: abs(x[0]))[:5]


# In[ ]:


# הגדרת המודל עם Cross-Validation
model = ElasticNetCV(alphas=np.logspace(-4, 0, 50), l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95], cv=10, random_state=42)
model.fit(X_train_reduced, y_train)

import pickle

# נניח שהמודל שלך מאומן ויש לך אותו באובייקט שנקרא `model`
# שמירת המודל כקובץ PKL
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# חישוב RMSE לכל קיפול ב-Cross-Validation
mse_path = model.mse_path_
mean_mse = mse_path.mean(axis=1)  # ממוצע ה-MSE על פני הקיפולים
best_rmse = np.sqrt(mean_mse.min())  # המרת MSE ל-RMSE והצגת ה-RMSE הטוב ביותר

print(f"Best RMSE from CV: {best_rmse}")

# חיזוי על סט הבדיקה
y_pred = model.predict(X_test_reduced)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# הצגת RMSE ותכונות עם המשקל הכי גבוה
print(f"Test RMSE: {test_rmse}")
print("Top 5 features:")
print(sorted(zip(model.coef_, [X_train.columns[i] for i in selected_features]), reverse=True, key=lambda x: abs(x[0]))[:5])

