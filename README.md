# 🎓 AP EAPCET College Prediction & Recommendation System

A premium, machine-learning-powered web application built using **Flask** and **scikit-learn** to predict college admissions for students based on their AP EAPCET rank, caste category, gender, and preferred districts/branches.

The system is powered by a **Random Forest Regressor** trained on **25,505 historical admission records** from the 2024 MPC Stream last rank cutoffs, providing a highly reliable estimate of admission bounds and filling in missing cutoff data automatically.

---

## 🌟 Key Features

1. **AI-Powered Predictions**: 
   - Uses a Random Forest Regressor to predict the expected cutoff rank for any combination of college, branch, district, caste, and gender.
   - Automatically imputes missing historical cutoffs (where no student from a specific category enrolled in 2024) to estimate your admission chance.
   
2. **Robust Input Filters**:
   - **Rank**: Enter your exact AP EAPCET rank.
   - **Caste Reservation Dropdown**: Supports all AP reservation groups: `OC`, `OC-EWS`, `BC-A`, `BC-B`, `BC-C`, `BC-D`, `BC-E`, `SC`, and `ST`.
   - **Gender Selection**: Supports specific cutoffs for `Male` (BOYS) and `Female` (GIRLS).
   - **District Multi-select (Optional)**: Choose one, multiple, or all of the **13 districts** in Andhra Pradesh using full names. If unchecked, it searches all districts.
   - **Branch Multi-select (Optional)**: Filter by any of the **69 engineering branches** showing their full names. If unchecked, it searches all branches. Includes a search box to filter branches by name.

3. **Dynamic Results Display**:
   - Classifies matches into three clear categories:
     - 🟢 **Safe (High Chance)**: Cutoff Rank $\ge$ 1.2 * User Rank.
     - 🟡 **Moderate (Medium Chance)**: Cutoff Rank between 0.9 * User Rank and 1.2 * User Rank.
     - 🔴 **Dream (Low Chance)**: Cutoff Rank between 0.7 * User Rank and 0.9 * User Rank.
   - **Search & Sort**: Filter colleges instantly by code, name, place, district, or branch, and click column headers to sort results dynamically.

---

## 🛠️ Technology Stack

* **Backend**: Flask (Python 3.x)
* **Machine Learning**: scikit-learn (Random Forest Regressor)
* **Data Processing**: pandas, numpy, openpyxl, xlrd
* **Frontend**: HTML5, CSS3 (Inter Google Font, Responsive Card Layout), JavaScript (Fetch API / AJAX)

---

## 🚀 Installation & Running Locally

### 1. Clone the Repository
```bash
git clone https://github.com/manoharreddy-1/College-Recommendation-System.git
cd College-Recommendation-System
```

### 2. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install flask pandas numpy scikit-learn xlrd openpyxl
```

### 3. Run the Server
Launch the Flask application:
```bash
python main.py
```

### 4. Open the Interface
Open your web browser and go to:
```url
http://127.0.0.1:5000
```
*(If port 5000 is occupied, the application will automatically fall back to port 5001).*
