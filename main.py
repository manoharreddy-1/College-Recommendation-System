import os
import sys
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Initialize Flask
app = Flask(__name__)

from flask.json.provider import DefaultJSONProvider

class CustomJSONProvider(DefaultJSONProvider):
    def default(self, o):
        if isinstance(o, (np.int64, np.int32, np.integer)):
            return int(o)
        if isinstance(o, (np.float64, np.float32, np.floating)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

app.json = CustomJSONProvider(app)

# Dataset path
DATASET_PATH = "APEAMCET2024LASTRANKDETAILSNONSW (1).XLS"
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = r"C:\Users\manoh\OneDrive\Desktop\aimpact\APEAMCET2024LASTRANKDETAILSNONSW (1).XLS"

# 13 Districts Mapping
DISTRICT_MAP = {
    'EG': 'East Godavari',
    'GTR': 'Guntur',
    'KRI': 'Krishna',
    'PKS': 'Prakasam',
    'SKL': 'Srikakulam',
    'VSP': 'Visakhapatnam',
    'VZM': 'Vizianagaram',
    'WG': 'West Godavari',
    'ATP': 'Anantapur',
    'CTR': 'Chittoor',
    'KDP': 'Kadapa',
    'KNL': 'Kurnool',
    'NLR': 'Nellore'
}

# Caste Categories Mapping
CASTE_MAP = {
    "OC": "OC",
    "OC-EWS": "OC_EWS",
    "BC-A": "BCA",
    "BC-B": "BCB",
    "BC-C": "BCC",
    "BC-D": "BCD",
    "BC-E": "BCE",
    "SC": "SC",
    "ST": "ST"
}

# Gender Mapping
GENDER_MAP = {
    "Male": "BOYS",
    "Female": "GIRLS"
}

# 69 Branches Full Names Mapping
BRANCH_FULL_NAMES = {
    'CIV': 'Civil Engineering',
    'CSE': 'Computer Science and Engineering',
    'ECE': 'Electronics and Communication Engineering',
    'EEE': 'Electrical and Electronics Engineering',
    'MEC': 'Mechanical Engineering',
    'CSD': 'Computer Science and Design',
    'CSM': 'Computer Science and Engineering (AI & ML)',
    'INF': 'Information Technology',
    'PHD': 'Pharm.D (Doctor of Pharmacy)',
    'EIE': 'Electronics and Instrumentation Engineering',
    'AIM': 'Artificial Intelligence and Machine Learning',
    'CAD': 'Computer Science and Engineering (AI & DS)',
    'AID': 'Artificial Intelligence and Data Science',
    'AGR': 'Agricultural Engineering',
    'CAI': 'Computer Science and Engineering (AI)',
    'DS': 'Computer Science and Engineering (Data Science)',
    'FDE': 'Food Engineering',
    'PHM': 'B.Pharmacy',
    'CHE': 'Chemical Engineering',
    'PEE': 'Petroleum Engineering',
    'CSC': 'Computer Science and Engineering (Cyber Security)',
    'CS': 'Computer Science',
    'CIT': 'Computer Science and Information Technology',
    'AUT': 'Automobile Engineering',
    'CSG': 'Computer Science and Engineering (IoT, CS & Block Chain)',
    'EVT': 'Environmental Engineering',
    'CSB': 'Computer Science and Business Systems',
    'CSO': 'Computer Science and Engineering (IoT)',
    'IOT': 'Internet of Things',
    'CIC': 'Computer Science and Engineering (IoT & CS)',
    'CBA': 'Computer Science and Business Systems (CBA)',
    'ECA': 'Electronics and Communication Engineering (Advanced)',
    'EII': 'Electronics and Instrumentation Engineering (Instrumentation)',
    'ASE': 'Aerospace Engineering',
    'CSER': 'Computer Science and Engineering (Robotics)',
    'MIN': 'Mining Engineering',
    'AI': 'Artificial Intelligence',
    'BIO': 'Biotechnology',
    'GIN': 'Geo-Informatics',
    'IST': 'Instrumentation Technology',
    'MET': 'Metallurgical Engineering',
    'NAM': 'Naval Architecture and Marine Engineering',
    'MRB': 'Marine Engineering',
    'ECM': 'Electronics and Computer Engineering',
    'CSS': 'Computer Systems and Software',
    'CST': 'Computer Science and Technology',
    'ECT': 'Electronics and Control Technology',
    'CSEB': 'Computer Science and Engineering (Big Data)',
    'RBT': 'Robotics and Automation',
    'FDT': 'Food Technology',
    'CCC': 'Cyber Security',
    'CIA': 'Cyber Security and IoT',
    'MMT': 'Metallurgical and Materials Engineering',
    'BDT': 'Big Data Technology',
    'PET': 'Petrochemical Engineering',
    'CBC': 'Computer Science and Engineering (Block Chain)',
    'CDA': 'Computer Science and Engineering (Data Science) - CDA',
    'CSW': 'Computer Science and Engineering (Cyber Security) - CSW',
    'ECES': 'Electronics and Communication Engineering (Sensors)',
    'ECV': 'Electronics and Communication Engineering (VLSI)',
    'MAD': 'Mobile Application Development',
    'SWE': 'Software Engineering',
    'GDT': 'GIS and Drone Technology',
    'CN': 'Computer Networks',
    'CSBS': 'Computer Science and Business Systems (CSBS)',
    'EBM': 'Electronics and Biomedical Engineering',
    'MAU': 'Mechanical and Automation Engineering',
    'MII': 'Manufacturing Engineering',
    'MMM': 'Materials and Metallurgical Engineering'
}

# Globally shared datasets and models
df_global = None
rank_cols_global = None
model_global = None
encoders_global = {}

def load_and_preprocess_data():
    global df_global, rank_cols_global
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")
    
    # Read Excel, skipping the decorative title row
    df = pd.read_excel(DATASET_PATH, header=1)
    df.columns = df.columns.str.strip()
    
    # Clean district values
    df['DIST'] = df['DIST'].str.strip()
    df['DIST_NAME'] = df['DIST'].map(DISTRICT_MAP).fillna(df['DIST'])
    
    # Clean branch values
    df['branch_code'] = df['branch_code'].str.strip()
    
    # Clean college names
    df['NAME OF THE INSTITUTION'] = df['NAME OF THE INSTITUTION'].str.strip()
    df['INSTCODE'] = df['INSTCODE'].str.strip()
    
    # Clean fees
    df['COLLFEE'] = pd.to_numeric(df['COLLFEE'], errors='coerce').fillna(43000).astype(int)
    
    # Convert rank cutoff columns to numeric
    rank_cols = [
        'OC_BOYS', 'OC_GIRLS', 'SC_BOYS', 'SC_GIRLS', 'ST_BOYS', 'ST_GIRLS',
        'BCA_BOYS', 'BCA_GIRLS', 'BCB_BOYS', 'BCB_GIRLS', 'BCC_BOYS', 'BCC_GIRLS',
        'BCD_BOYS', 'BCD_GIRLS', 'BCE_BOYS', 'BCE_GIRLS', 'OC_EWS_BOYS', 'OC_EWS_GIRLS'
    ]
    for col in rank_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df_global = df
    rank_cols_global = rank_cols

def train_prediction_model():
    global model_global, encoders_global
    # Melt dataset into long format for model training
    melted = df_global.melt(
        id_vars=['INSTCODE', 'NAME OF THE INSTITUTION', 'TYPE', 'DIST', 'AFFL.', 'branch_code', 'COLLFEE'],
        value_vars=rank_cols_global,
        var_name='Category',
        value_name='CutoffRank'
    ).dropna(subset=['CutoffRank'])
    
    # Extract Caste and Gender
    def parse_category(c):
        if c.endswith('_BOYS'):
            return c[:-5], 'BOYS'
        return c[:-6], 'GIRLS'
        
    cats = melted['Category'].apply(parse_category)
    melted['Caste'] = [x[0] for x in cats]
    melted['Gender'] = [x[1] for x in cats]
    
    # Features & Targets
    features = ['INSTCODE', 'branch_code', 'DIST', 'TYPE', 'AFFL.', 'Caste', 'Gender']
    X = melted[features + ['COLLFEE']].copy()
    
    # Encode categorical variables
    encoders_global = {}
    for col in features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders_global[col] = le
        
    y = melted['CutoffRank']
    
    # Fit Random Forest Regressor
    model = RandomForestRegressor(n_estimators=40, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X, y)
    model_global = model

# Load and train immediately upon importing/starting
print("Loading data...")
load_and_preprocess_data()
print("Training AI Model...")
train_prediction_model()
print("AI Model Trained successfully.")

# Support dry-runs for validation scripts
if '--dry-run' in sys.argv:
    print("Dry-run verification completed successfully.")
    sys.exit(0)

@app.route('/')
def index():
    # Sort districts and branches by name for display
    sorted_districts = {k: DISTRICT_MAP[k] for k in sorted(DISTRICT_MAP.keys(), key=lambda x: DISTRICT_MAP[x])}
    sorted_branches = {k: BRANCH_FULL_NAMES[k] for k in sorted(BRANCH_FULL_NAMES.keys(), key=lambda x: BRANCH_FULL_NAMES[x])}
    
    return render_template(
        'index.html',
        districts=sorted_districts,
        branches=sorted_branches
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        rank = int(data.get('rank', 25000))
    except (ValueError, TypeError):
        rank = 25000
    caste = data.get('caste', 'OC')
    gender = data.get('gender', 'Male')
    districts = data.get('districts', [])
    branches = data.get('branches', [])
    
    caste_code = CASTE_MAP.get(caste, 'OC')
    gender_code = GENDER_MAP.get(gender, 'BOYS')
    target_rank_col = f"{caste_code}_{gender_code}"
    
    # Apply basic location/branch filters
    filtered = df_global.copy()
    if districts:
        filtered = filtered[filtered['DIST'].isin(districts)]
    if branches:
        filtered = filtered[filtered['branch_code'].isin(branches)]
    
    if filtered.empty:
        return jsonify([])
        
    # Generate ML Predictions
    test_X = pd.DataFrame()
    for col in ['INSTCODE', 'branch_code', 'DIST', 'TYPE', 'AFFL.']:
        le = encoders_global[col]
        values = filtered[col].astype(str).tolist()
        encoded_vals = []
        for val in values:
            if val in le.classes_:
                encoded_vals.append(le.transform([val])[0])
            else:
                encoded_vals.append(0)
        test_X[col] = encoded_vals
        
    test_X['Caste'] = encoders_global['Caste'].transform([caste_code] * len(filtered))
    test_X['Gender'] = encoders_global['Gender'].transform([gender_code] * len(filtered))
    test_X['COLLFEE'] = filtered['COLLFEE'].fillna(43000).values
    
    # Predict cutoffs using RandomForest model
    ml_cutoffs = model_global.predict(test_X).astype(int)
    
    # Compile output data list
    results = []
    for idx, (_, row) in enumerate(filtered.iterrows()):
        actual_cutoff = row[target_rank_col]
        is_imputed = pd.isna(actual_cutoff)
        effective_cutoff = int(ml_cutoffs[idx]) if is_imputed else int(actual_cutoff)
        
        # Only show colleges where the student has a chance (Effective Cutoff >= 0.7 * rank)
        if effective_cutoff >= 0.7 * rank:
            # Categorize admission chance
            ratio = effective_cutoff / rank
            if ratio >= 1.2:
                chance = "Safe (High Chance)"
                chance_class = "safe"
            elif ratio >= 0.9:
                chance = "Moderate (Medium Chance)"
                chance_class = "moderate"
            else:
                chance = "Dream (Low Chance)"
                chance_class = "dream"
                
            results.append({
                "college_code": row['INSTCODE'],
                "college_name": row['NAME OF THE INSTITUTION'],
                "district": row['DIST_NAME'],
                "place": row['PLACE'],
                "branch": BRANCH_FULL_NAMES.get(row['branch_code'], row['branch_code']),
                "cutoff_actual": None if is_imputed else int(actual_cutoff),
                "cutoff_predicted": int(ml_cutoffs[idx]),
                "ai_imputed": is_imputed,
                "chance": chance,
                "chance_class": chance_class,
                "effective_cutoff": effective_cutoff
            })
            
    # Sort results by effective cutoff ascending (closest / hardest first)
    results = sorted(results, key=lambda x: x['effective_cutoff'])
    return jsonify(results)

if __name__ == '__main__':
    port = 5000
    try:
        app.run(debug=False, host='0.0.0.0', port=port)
    except Exception as e:
        if "10048" in str(e) or "Address already in use" in str(e):
            print(f"\n⚠️ Port {port} is currently in use by another running python process.")
            print(f"Attempting to start the server on port 5001 instead...")
            app.run(debug=False, host='0.0.0.0', port=5001)
        else:
            raise e
