import joblib
import pandas as pd
import numpy as np

# 1. Load the "Brain"
model = joblib.load('models/propiq_model.pkl')

def get_valuation(sqft, qual, baths, age, neighborhood):
    # Create a small dataframe for the model
    data = pd.DataFrame({
        'totalsqft': [sqft],
        'overallqual': [qual],
        'totalbaths': [baths],
        'houseage': [age],
        'neighborhood': [neighborhood],
        'grlivarea': [sqft * 0.7] # Approximation for living area
    })
    
    # Predict (results will be in Log, so we use expm1 to get dollars)
    log_prediction = model.predict(data)
    price = np.expm1(log_prediction)[0]
    
    print(f"\n🏠 PropIQ Property Valuation")
    print(f"-------------------------------")
    print(f"Neighborhood: {neighborhood}")
    print(f"Quality:      {qual}/10")
    print(f"Total Size:   {sqft} sqft")
    print(f"-------------------------------")
    print(f"ESTIMATED VALUE: ${price:,.2f}")

# TEST IT: A high-quality, large house in 'NoRidge' (Northridge)
get_valuation(sqft=3000, qual=9, baths=3, age=5, neighborhood='NoRidge')