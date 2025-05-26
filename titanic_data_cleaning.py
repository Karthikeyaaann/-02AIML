#!/usr/bin/env python3
"""
Titanic Dataset Cleaning Pipeline
Input: Raw Titanic dataset (CSV)
Output: Cleaned dataset ready for analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import argparse

def load_data(filepath):
    """Load dataset with error handling"""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Successfully loaded data (shape: {df.shape})")
        return df
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None

def clean_data(df):
    """
    Main cleaning pipeline with documented steps
    """
    # 1. Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # 2. Drop unnecessary columns
    cols_to_drop = ['Cabin', 'PassengerId', 'Name', 'Ticket']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # 3. Encode categorical variables
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True, prefix='Emb')
    
    # 4. Scale numerical features
    scaler = StandardScaler()
    if 'Age' in df.columns and 'Fare' in df.columns:
        df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
    
    # 5. Remove outliers (IQR method)
    if 'Fare' in df.columns:
        Q1 = df['Fare'].quantile(0.25)
        Q3 = df['Fare'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df['Fare'] >= Q1 - 1.5*IQR) & (df['Fare'] <= Q3 + 1.5*IQR)]
    
    return df

def save_data(df, output_path):
    """Save cleaned data with validation"""
    try:
        df.to_csv(output_path, index=False)
        print(f"💾 Cleaned data saved to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to save data: {e}")
        return False

def main():
    # Configure command-line interface
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/raw/titanic.csv', help='Input file path')
    parser.add_argument('--output', default='data/processed/cleaned_titanic.csv', help='Output file path')
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Execute pipeline
    print("\n🚀 Starting Titanic data cleaning pipeline")
    df = load_data(args.input)
    if df is not None:
        cleaned_df = clean_data(df)
        if save_data(cleaned_df, args.output):
            print("✨ Pipeline completed successfully!")
        else:
            print("⚠️ Completed with save error")
    else:
        print("🛑 Pipeline failed")

if __name__ == "__main__":
    main()