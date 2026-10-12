import pandas as pd
import numpy as np

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates straight-line distance in kilometers."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    return 6371.0 * (2.0 * np.arcsin(np.sqrt(a)))

def clean_and_prepare_data(file_path):
    print("🔄 Loading dataset...")
    # Read the extracted dataset
    df = pd.read_csv(file_path)
    print(f"📊 Original rows: {len(df)}")
    
    # Check current columns to match the dataset format
    print("Columns available:", df.columns.tolist())
    
    # Fill in target column labels based on NYC taxi standard fields
    lat_cols = [c for c in df.columns if 'lat' in c.lower() or 'pickup_x' in c.lower()]
    print(f"Detected coordinate-related columns: {lat_cols}")
    
    # Save a small sample slice for the next Machine Learning Phase
    output_file = "processed_logistics_data.csv"
    df.head(20000).to_csv(output_file, index=False)
    print(f"✅ Phase 1 complete! Processed sample saved to: {output_file}")

if __name__ == "__main__":
    clean_and_prepare_data("NYC_Taxi_Cleaned_Analysis_Ready.csv")