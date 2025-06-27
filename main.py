import pandas as pd
import warnings
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Suppress warnings
warnings.filterwarnings("ignore")

# Load dataset
file_path = r"C:\Users\manoh\OneDrive\Desktop\aimpact\filtered_college_recommendations.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Select relevant features
features = ['Academic Score', 'Preferred Location', 'Sports Interest',
            'Placement Preference', 'Extracurricular Activities', 'Preferred Department']
X = df[features]
y = df['College Recommendation']

# Encode categorical variables
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode target variable
y_le = LabelEncoder()
y = y_le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Streamlit UI
st.title("College Recommendation System")

# User inputs
user_input = {}
for feature in features:
    if df[feature].dtype == 'O':
        user_input[feature] = st.selectbox(f"Select {feature}", df[feature].unique())
    else:
        user_input[feature] = st.number_input(f"Enter {feature}", min_value=0, max_value=100, step=1)

# Predict button
if st.button("Recommend College"):
    input_df = pd.DataFrame([user_input], columns=features)
    for col in input_df.select_dtypes(include=['object']).columns:
        input_df[col] = label_encoders[col].transform(input_df[col])
    predicted_college = y_le.inverse_transform(model.predict(input_df))[0]
    st.success(f"Recommended College: {predicted_college}")
