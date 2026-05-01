import pandas as pd
import numpy as np
import sqlite3
import os

def run_feature_engineering():
    print("🚀 Starting Feature Engineering...")
    
    # 1. Load Data from your 'Warehouse'
    conn = sqlite3.connect('data/propiq.db')
    df = pd.read_sql_query("SELECT * FROM listings", conn)
    conn.close()
    
    # 2. Standardize Names (Matches your successful EDA flow)
    df.columns = [col.lower().replace(' ', '').replace('_', '') for col in df.columns]
    
    # 3. Handle 'None' Imputation
    # For these columns, 'NaN' means the feature doesn't exist (e.g., No Pool)
    none_cols = ['poolqc', 'miscfeature', 'alley', 'fence', 'fireplacequ', 
                 'garagetype', 'garagefinish', 'garagequal', 'garagecond',
                 'bsmtqual', 'bsmtcond', 'bsmtexposure', 'bsmtfintype1', 'bsmtfintype2']
    
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')

    # 4. Handle Missing Numerical Values
    # If Lot Frontage is missing, we'll use the median of the neighborhood
    if 'lotfrontage' in df.columns:
        df['lotfrontage'] = df['lotfrontage'].fillna(df['lotfrontage'].median())
    
    # 5. Feature Creation: The "Big Three"
    # Total SF: Combining all living areas into one powerhouse feature
    df['totalsqft'] = df['1stflrsf'] + df['2ndflrsf'] + df['totalbsmtsf']
    
    # Total Bathrooms: (Full + 0.5 * Half)
    df['totalbaths'] = df['fullbath'] + (0.5 * df['halfbath']) + df['bsmtfullbath'] + (0.5 * df['bsmthalfbath'])
    
    # House Age at time of sale
    df['houseage'] = df['yrsold'] - df['yearbuilt']

    # 6. Save Processed Data
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/cleaned_listings.csv', index=False)
    
    print(f"✅ Success! Cleaned file saved to data/processed/cleaned_listings.csv")
    print(f"📊 New Features Created: totalsqft, totalbaths, houseage")

if __name__ == "__main__":
    run_feature_engineering()