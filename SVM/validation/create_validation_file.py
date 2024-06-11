import pandas as pd

def create_validation_data(input_csv, validation_csv, sample_size=100):
    # Read the original data
    df = pd.read_csv(input_csv)
    
    # Sample a subset of the data for validation (or you can use any selection criteria)
    validation_df = df.sample(n=sample_size, random_state=42)
    
    # Select relevant columns
    validation_data = validation_df[['artist', 'track', 'text']]
    
    # Save to a new CSV file
    validation_data.to_csv(validation_csv, index=False)
    
if __name__ == "__main__":
    input_csv = "complete_data.csv"
    validation_csv = "validation_data.csv"
    
    create_validation_data(input_csv, validation_csv)
    print(f"Validation data saved to {validation_csv}")
