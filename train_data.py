from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

# Load dataset
data = pd.read_csv(r"C:\Users\REINO AVANSATRIA\Downloads\Medicio\Medicio\breast-cancer-wisconsin-data_data.csv")

# Drop unnecessary columns
data = data.drop(columns=['Unnamed: 32', 'id'])

# Encode target variable
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Select the 4 features
selected_features = [
    'area_worst', 'perimeter_worst', 'radius_worst', 'concave points_worst'
]
X = data[selected_features]
y = data['diagnosis']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model and scaler
joblib.dump(model, 'cancer_model_4_features.pkl')
joblib.dump(scaler, 'scaler_4_features.pkl')

print("Model and scaler saved successfully.")
