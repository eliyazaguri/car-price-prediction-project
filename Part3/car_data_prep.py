#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# איחוד כל המופעים של Lexsus ו- לקסוס לשם אחיד
def clean_manufactor_names(df):
    df['manufactor'] = df['manufactor'].replace(['Lexsus', 'לקסוס'], 'לקסוס')
    return df


# In[2]:


def add_important_words_count(df):
    # רשימת המילים שיכולות להעלות את ערך הרכב
    important_words = ["כחדש", "שמור", "שמורה", "ללא תאונות","מתוחזק","מקורה","נהג יחיד","חדש","אחרי טיפול"
                       ,"מגנזיום","מורשה","חסכוני","מובילאיי","בבדיקה","מערכת"]

    # פונקציה פנימית לספירת המילים החשובות בתיאור
    def count_important_words(Description, words):
        if not Description or Description.strip() == '':
            return 0
        count = 0
        for word in words:
            if word in Description:
                count += 1
        return count

    # יצירת העמודה החדשה עם ספירת המילים החשובות
    df['important_words_count'] = df['Description'].apply(lambda x: count_important_words(str(x), important_words))
    df['important_words_count'] = df['important_words_count'] * 100
    
    return df


# In[ ]:


def fill_missing_gear(df):
    # מציאת הערך הנפוץ ביותר בעמודת Gear
    most_common_gear = df['Gear'].mode()[0]

    # מילוי הערך החסר בעמודת Gear עם הערך הנפוץ ביותר
    df['Gear'] = df['Gear'].fillna(most_common_gear)
    
    return df


# In[ ]:


def fill_missing_capacity_and_engine_type(df):
    # שלב 1: יצירת מיפוי לערך הנפוץ ביותר עבור כל שילוב של יצרן ודגם עבור capacity_Engine ו- Engine_type
    capacity_mapping = df.groupby(['manufactor', 'model'])['capacity_Engine'].apply(
        lambda x: x.mode()[0] if not x.mode().empty else None
    )

    engine_type_mapping = df.groupby(['manufactor', 'model'])['Engine_type'].apply(
        lambda x: x.mode()[0] if not x.mode().empty else None
    )

    # שלב 2: פונקציות להשלמת ערכים חסרים עבור capacity_Engine ו- Engine_type לפי יצרן ודגם
    def fill_capacity_engine(row):
        if pd.isna(row['capacity_Engine']):
            return capacity_mapping.get((row['manufactor'], row['model']), row['capacity_Engine'])
        return row['capacity_Engine']

    def fill_engine_type(row):
        if pd.isna(row['Engine_type']):
            return engine_type_mapping.get((row['manufactor'], row['model']), row['Engine_type'])
        return row['Engine_type']

    # שלב 3: החלת הפונקציות על העמודות
    df['capacity_Engine'] = df.apply(fill_capacity_engine, axis=1)
    df['Engine_type'] = df.apply(fill_engine_type, axis=1)

    # שלב 4: יצירת מיפוי לערך השכיח ביותר עבור כל יצרן עבור capacity_Engine ו- Engine_type
    capacity_mapping_by_manufactor = df.groupby('manufactor')['capacity_Engine'].apply(
        lambda x: x.mode()[0] if not x.mode().empty else None
    )

    engine_type_mapping_by_manufactor = df.groupby('manufactor')['Engine_type'].apply(
        lambda x: x.mode()[0] if not x.mode().empty else None
    )

    # שלב 5: פונקציות להשלמת ערכים חסרים לפי היצרן
    def fill_capacity_engine_by_manufactor(row):
        if pd.isna(row['capacity_Engine']):
            return capacity_mapping_by_manufactor.get(row['manufactor'], row['capacity_Engine'])
        return row['capacity_Engine']

    def fill_engine_type_by_manufactor(row):
        if pd.isna(row['Engine_type']):
            return engine_type_mapping_by_manufactor.get(row['manufactor'], row['Engine_type'])
        return row['Engine_type']

    # שלב 6: החלת הפונקציות על העמודות
    df['capacity_Engine'] = df.apply(fill_capacity_engine_by_manufactor, axis=1)
    df['Engine_type'] = df.apply(fill_engine_type_by_manufactor, axis=1)

    # בדיקה של מספר הערכים החסרים לאחר ההשלמה
    missing_capacity = df['capacity_Engine'].isnull().sum()
    missing_engine_type = df['Engine_type'].isnull().sum()

    print(f"Missing capacity_Engine values: {missing_capacity}")
    print(f"Missing Engine_type values: {missing_engine_type}")

    return df


# In[ ]:


def process_ownership_columns(df):
    # הצגת הערכים הייחודיים בעמודת Prev_ownership וספירתם
    prev_ownership_values = df['Prev_ownership'].value_counts(dropna=False)
    print("Prev_ownership values:")
    print(prev_ownership_values)

    # הצגת הערכים הייחודיים בעמודת Curr_ownership וספירתם
    curr_ownership_values = df['Curr_ownership'].value_counts(dropna=False)
    print("\nCurr_ownership values:")
    print(curr_ownership_values)
    
    # שלב 1: הגדרת תנאי האיחוד
    def unify_ownership(value):
        if pd.isna(value) or value in ["לא מוגדר", "אחר", "None"]:
            return ""  # איחוד לערך ריק
        elif value in ["חברה", "השכרה", "ליסינג", "ממשלתי"]:
            return "השכרה"  # איחוד לערך "השכרה"
        else:
            return value  # החזרת הערך המקורי אם לא שייך לאחת הקטגוריות הנ"ל

    # שלב 2: החלת הפונקציה על עמודות Prev_ownership ו- Curr_ownership
    df['Prev_ownership'] = df['Prev_ownership'].apply(unify_ownership)
    df['Curr_ownership'] = df['Curr_ownership'].apply(unify_ownership)

    # הצגת הערכים המעודכנים לאחר האיחוד
    print("Prev_ownership values after unification:")
    print(df['Prev_ownership'].value_counts(dropna=False))

    print("\nCurr_ownership values after unification:")
    print(df['Curr_ownership'].value_counts(dropna=False))
    
    # מילוי ערכים ריקים או חסרים בעמודת Prev_ownership בערך "לא ידוע"
    df['Prev_ownership'] = df['Prev_ownership'].replace("", "לא ידוע").fillna("לא ידוע")

    # מילוי ערכים ריקים או חסרים בעמודת Curr_ownership בערך "לא ידוע"
    df['Curr_ownership'] = df['Curr_ownership'].replace("", "לא ידוע").fillna("לא ידוע")

    # הצגת הערכים המעודכנים לאחר המילוי
    print("Prev_ownership values after update:")
    print(df['Prev_ownership'].value_counts(dropna=False))

    print("\nCurr_ownership values after update:")
    print(df['Curr_ownership'].value_counts(dropna=False))
    
    return df


# In[ ]:


def fill_missing_area_with_city(df):
    # מילוי ערכים ריקים או חסרים בעמודת AREA בשם העיר מעמודת City
    df['Area'] = df.apply(lambda row: row['City'] if pd.isna(row['Area']) or row['Area'] == "" else row['Area'], axis=1)
    
    # הצגת הערכים המעודכנים לאחר המילוי
    print("Area values after update:")
    print(df['Area'].value_counts(dropna=False))
    
    return df


# In[ ]:


def fill_missing_pic_num(df):
    # מילוי ערכים ריקים או חסרים בעמודת Pic_num בערך 0
    df['Pic_num'] = df['Pic_num'].replace("", 0).fillna(0)
    
    # הצגת הערכים המעודכנים לאחר המילוי
    print("Pic_num values after update:")
    print(df['Pic_num'].value_counts(dropna=False))
    
    return df


# In[ ]:


import numpy as np

def fill_missing_color(df):
    # איחוד של הערכים NaN ו-None לערך אחיד (NaN במקרה זה)
    df['Color'] = df['Color'].replace("None", np.nan)
    
    # הצגת רשימת כל הצבעים הייחודיים לאחר האיחוד
    color_values = df['Color'].value_counts(dropna=False)
    print("Color values before filling from description:")
    print(color_values)
    
    # יצירת רשימת הצבעים הקיימים
    color_list = [
        "לבן", "שחור", "אפור מטאלי", "אפור", "כסוף", "לבן פנינה", "אפור עכבר", "כחול", "כחול כהה מטאלי", 
        "לבן שנהב", "כסוף מטאלי", "לבן מטאלי", "כסף מטלי", "אדום", "זהב מטאלי", "כחול כהה", "בז' מטאלי", 
        "תכלת", "חום", "ירוק בהיר", "סגול חציל", "שמפניה", "תכלת מטאלי", "ירוק", "כחול בהיר", "חום מטאלי", 
        "אדום מטאלי", "בז'", "בורדו", "טורקיז", "כחול מטאלי", "כחול בהיר מטאלי", "ברונזה", "סגול", 
        "ירוק מטאלי", "כתום", "זהב", "ורוד"
    ]

    # פונקציה למילוי הצבע על פי התיאור
    def fill_color_from_description(row):
        if pd.isna(row['Color']):  # בדיקה אם הצבע חסר (NaN)
            for color in color_list:
                if color in row['Description']:
                    return color
        return row['Color']

    # החלת הפונקציה על כל השורות עם ערכים חסרים בעמודת Color
    df['Color'] = df.apply(fill_color_from_description, axis=1)

    # החלפת ערכי NaN בערך "לא ידוע" בעמודת Color
    df['Color'] = df['Color'].fillna("לא ידוע")

    # הצגת הערכים המעודכנים לאחר ההחלפה
    print("Color values after update:")
    print(df['Color'].value_counts(dropna=False))
    
    return df


# In[7]:


import pandas as pd
import numpy as np

def clean_and_predict_km(df):
    # ניקוי עמודת Capacity_Engine כך שכל הערכים יומרו למספרים
    df['capacity_Engine'] = df['capacity_Engine'].replace(',', '', regex=True).astype(float)

    # ניקוי עמודת Km כדי להמיר מחרוזות למספרים, כולל טיפול בערכים לא חוקיים
    df['Km'] = df['Km'].replace(',', '', regex=True)
    df['Km'] = pd.to_numeric(df['Km'], errors='coerce')

    # מציאת ערכים חסרים בעמודת Km
    missing_km_indices = df[df['Km'].isna()].index

    # מילוי הערכים החסרים ב-Km על בסיס גיל הרכב והטווח 13,000-15,000 ק"מ בשנה
    df.loc[missing_km_indices, 'Km'] = df.loc[missing_km_indices, 'Year'].apply(
        lambda year: np.random.randint(13000, 15000) * (2024 - year)
    )
    
    return df


# In[ ]:


import pandas as pd
import numpy as np
from datetime import datetime

def process_test_column(df):
    # פונקציה להמרת תאריכים בפורמט "MMM-YY" לתאריך מלא
    def convert_partial_date(date_str):
        try:
            return pd.to_datetime(date_str, format='%b-%y')
        except:
            return pd.NaT

    # נסה להמיר תאריכים רגילים
    df['Test_as_date'] = pd.to_datetime(df['Test'], errors='coerce', dayfirst=True)

    # המרה נוספת של תאריכים בפורמט "MMM-YY"
    df['Test_as_date'] = df['Test_as_date'].combine_first(df['Test'].apply(convert_partial_date))

    # חישוב מספר הימים עד הטסט הבא מהתאריך 1 ביוני 2024
    reference_date = datetime(2024, 6, 1)
    df['Days_until_test'] = (df['Test_as_date'] - reference_date).dt.days

    # עדכון הערכים בעמודת Test רק עבור תאריכים
    df['Test'] = np.where(df['Test_as_date'].notna(), df['Days_until_test'], df['Test'])

    # הסרת עמודות העזר
    df = df.drop(columns=['Test_as_date', 'Days_until_test'])

    # החלפת ערכי 'None' ב-NaN
    df['Test'] = df['Test'].replace('None', np.nan)

    # המרת ערכים שליליים לערכים חיוביים
    df['Test'] = df['Test'].apply(lambda x: abs(x) if isinstance(x, (int, float)) and x < 0 else x)

    # עיגול ערכים עשרוניים לערכים שלמים
    df['Test'] = df['Test'].apply(lambda x: round(x) if isinstance(x, float) and not np.isnan(x) else x)

    # המרת ערכים מחרוזתיים למספרים (אם אפשרי)
    df['Test'] = pd.to_numeric(df['Test'], errors='coerce')

    # המרת ערכים שליליים לערכים חיוביים
    df['Test'] = df['Test'].apply(lambda x: abs(x) if pd.notna(x) and x < 0 else x)

    # מילוי הערכים החסרים (NaN) בערך של חצי שנה (182.5 ימים)
    df['Test'] = df['Test'].fillna(365 / 2)

    # הצגת התוצאות לאחר המילוי
    print("Test values after processing:")
    print(df['Test'].describe())
    
    return df


# In[ ]:


def fill_missing_supply_score(df):
    # חישוב הממוצע של Supply_score עבור כל יצרן
    manufacturer_mean_supply_score = df.groupby('manufactor')['Supply_score'].mean()

    # השלמת הערכים החסרים לפי הממוצע של היצרן המתאים
    df['Supply_score'] = df.apply(
        lambda row: manufacturer_mean_supply_score[row['manufactor']] if pd.isnull(row['Supply_score']) else row['Supply_score'],
        axis=1
    )

    # חישוב הממוצע הכללי של Supply_score
    overall_mean_supply_score = df['Supply_score'].mean()

    # השלמת הערכים החסרים שנותרו עם הממוצע הכללי
    df['Supply_score'].fillna(overall_mean_supply_score, inplace=True)

    # בדיקה של השלמת הערכים
    print("Missing values in Supply_score after processing:", df['Supply_score'].isnull().sum())
    
    return df


# In[ ]:


def create_km_per_year(df):
    
    # יצירת התכונה Km per Year
    df['Km_per_Year'] = df['Km'] / (df['Year'] - df['Year'].min() + 1)  # כדי למנוע חלוקה באפס

    # סינון נתונים עם ערכים לא חוקיים (אם ישנם)
    df = df.dropna(subset=['Km_per_Year'])

    return df


# In[ ]:


def create_car_age(df):
    df['Car_Age'] = 2024 - df['Year']
    return df


# In[ ]:


def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    
    return df


# In[ ]:


from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    # יצירת עמודת Car_Age והסרת Year
    df['Car_Age'] = 2024 - df['Year']
    
    # בחירת התכונות הרצויות
    selected_features = ['manufactor', 'Car_Age', 'Km_per_Year','Engine_type','model', 'Hand', 'capacity_Engine', 'important_words_count', 'Color', 'Km', 'Test', 'Supply_score']
    df = df.loc[:, selected_features]

    # קידוד תכונות קטגוריאליות עם Label Encoding
    label_encoders = {}
    for col in ['manufactor', 'model', 'Color','Engine_type']:
        le = LabelEncoder()
        df.loc[:, col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # נורמליזציה של התכונות המספריות
    numeric_features = ['Car_Age', 'Hand', 'capacity_Engine', 'Km_per_Year','Km', 'Test', 'Supply_score','important_words_count']
    scaler = StandardScaler()
    df.loc[:, numeric_features] = scaler.fit_transform(df[numeric_features])
    
    return df


# In[4]:


def prepare_data(df):
    # הפעלת כל הפונקציות אחת אחרי השנייה
    df = add_important_words_count(df)
    df = fill_missing_gear(df)
    df = fill_missing_capacity_and_engine_type(df)
    df = process_ownership_columns(df)
    df = fill_missing_area_with_city(df)
    df = fill_missing_pic_num(df)
    df = fill_missing_color(df)
    df = clean_and_predict_km(df)
    df = process_test_column(df)
    df = fill_missing_supply_score(df)
    df = create_km_per_year(df)
    df = create_car_age(df)
    df = cap_outliers(df,'Km')
    df = preprocess_data(df)
    
    return df

