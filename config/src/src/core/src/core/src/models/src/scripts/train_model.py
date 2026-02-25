#!/usr/bin/env python
"""Script to train the AI model for admission prediction."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_training_data(n_samples=1000):
    """Generate synthetic training data."""
    np.random.seed(42)
    
    data = []
    for i in range(n_samples):
        # Generate features
        gpa = np.random.uniform(2.0, 4.0)
        sat = np.random.randint(1000, 1600)
        gre = np.random.randint(290, 340)
        experience_years = np.random.randint(0, 8)
        statement_quality = np.random.uniform(0.3, 1.0)
        
        # Calculate score (simplified)
        academic_score = (gpa - 2.0) / 2.0  # Normalize to 0-1
        test_score = ((sat - 1000) / 600 + (gre - 290) / 50) / 2
        exp_score = min(experience_years / 5, 1.0)
        
        overall_score = (
            academic_score * 0.4 +
            test_score * 0.3 +
            exp_score * 0.15 +
            statement_quality * 0.15
        )
        
        # Generate target (admitted or not)
        if overall_score > 0.7:
            admitted = 1
        elif overall_score > 0.5:
            admitted = np.random.choice([0, 1], p=[0.3, 0.7])
        elif overall_score > 0.3:
            admitted = np.random.choice([0, 1], p=[0.7, 0.3])
        else:
            admitted = 0
        
        data.append({
            'gpa': gpa,
            'sat': sat,
            'gre': gre,
            'experience_years': experience_years,
            'statement_quality': statement_quality,
            'admitted': admitted
        })
    
    return pd.DataFrame(data)

def train_model():
    """Train the admission prediction model."""
    print("Generating training data...")
    df = generate_training_data(2000)
    
    # Prepare features and target
    feature_cols = ['gpa', 'sat', 'gre', 'experience_years', 'statement_quality']
    X = df[feature_cols]
    y = df['admitted']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(importance)
    
    # Save model
    model_path = "models/saved/admission_model.h5"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    return model

if __name__ == "__main__":
    train_model()
