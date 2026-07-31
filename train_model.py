import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Define columns
num_scaled_cols = ['ph', 'turbidity', 'conductivity', 'temperature']
other_cols = ['year', 'month', 'day', 'dayofweek', 'quarter']  # not scaled
feature_cols = other_cols + num_scaled_cols
target_col = 'coagulant_dosage_log'

# Load data
df = pd.read_csv("assets/water_quality_cleaned.csv")
print('Data loaded successfully')

X = df[feature_cols]
y = df[target_col]

# ColumnTransformer: scale only selected numeric columns
scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('scaled_nums', scaler, num_scaled_cols)
    ],
    remainder='passthrough'  # year/month stay as-is
)

model = XGBRegressor(random_state=10, n_jobs=-1, colsample_bytree=0.8, 
                     gamma=0, learning_rate=0.1, max_depth=3, n_estimators=100, subsample=0.8)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Time spliting
train_size = int(len(df) * 0.9) # 90% of dataset
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

# train model
pipeline.fit(X_train, y_train)
print('Model successfully trained')

joblib.dump(pipeline, "ml_pipeline.joblib")
print("Saved to ml_pipeline.joblib")