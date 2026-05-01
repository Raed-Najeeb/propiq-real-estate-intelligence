import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def train():
    print("🤖 Initializing Robust Model Training...")
    
    # 1. Load the cleaned data
    df = pd.read_csv('data/processed/cleaned_listings.csv')
    
    # 2. Select Features and Target
    features = ['totalsqft', 'overallqual', 'totalbaths', 'houseage', 'neighborhood', 'grlivarea']
    X = df[features]
    y = np.log1p(df['saleprice'])

    # 3. Split into Train/Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Define Preprocessing
    # Numeric: Fill NaNs with Median + Scale
    numeric_features = ['totalsqft', 'overallqual', 'totalbaths', 'houseage', 'grlivarea']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical: Fill NaNs with 'missing' + OneHot
    categorical_features = ['neighborhood']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine them
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 5. Create the Final Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
    ])

    # 6. Train
    model_pipeline.fit(X_train, y_train)

    # 7. Evaluate
    preds = model_pipeline.predict(X_test)
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(preds))
    r2 = r2_score(y_test, preds)

    print(f"✅ Training Complete!")
    print(f"📈 Model Accuracy (R²): {r2:.4f}")
    print(f"💸 Average Error (MAE): ${mae:,.2f}")

    # 8. Save
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_pipeline, 'models/propiq_model.pkl')
    print("💾 Model saved to models/propiq_model.pkl")

if __name__ == "__main__":
    train()