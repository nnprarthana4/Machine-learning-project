import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
import warnings

# Suppress scikit-learn warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the dataset
file_path = "D:/mL project/crop_yield.csv"
df = pd.read_csv(file_path)

# Handle missing values
df.dropna(inplace=True)

# Identify and encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Initialize OneHotEncoder
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

# Fit encoder and transform categorical columns
encoded_data = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
df.drop(columns=categorical_cols, inplace=True)
df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)

# Convert target Yield into classification labels
yield_median = df['Yield'].median()
df['Yield_Class'] = (df['Yield'] >= yield_median).astype(int)

# Drop original Yield column
df.drop(columns=['Yield'], inplace=True)

# Save cleaned dataset
df.to_csv("cleaned_crop_yield_classification.csv", index=False)

# Features and target
X = df.drop(columns=['Yield_Class'])
y = df['Yield_Class']

# Store feature names in a variable
feature_names = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
print("\n==== Random Forest ====")

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print metrics
print(f"Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred) * 100:.2f}%")
print(f"Recall: {recall_score(y_test, y_pred) * 100:.2f}%")
print(f"F1 Score: {f1_score(y_test, y_pred) * 100:.2f}%")

conf_matrix = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = conf_matrix.ravel()
print("Confusion Matrix:")
print(conf_matrix)
print(f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest - Confusion Matrix")
plt.show()

plt.figure(figsize=(10, 5))
sns.lineplot(x=list(range(1, 11)), y=list(cv_scores * 100), marker='o', linestyle='--', color='blue')
plt.xlabel("Fold")
plt.ylabel("Accuracy (%)")
plt.title("Random Forest - Cross-Validation Accuracy per Fold")
plt.grid(True)
plt.show()

# Get feature names of encoded columns
encoded_feature_names = encoder.get_feature_names_out(categorical_cols)

# Feature importance visualization
if hasattr(model, 'feature_importances_'):
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot the feature importances
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.show()

# Prediction functions
def preprocess_input_data(features, encoder, feature_names):
    categorical_data = features[list(encoder.feature_names_in_)]
    numerical_data = features.drop(columns=list(encoder.feature_names_in_))

    encoded_data = encoder.transform(categorical_data)

    all_feature_names = list(numerical_data.columns) + list(encoder.get_feature_names_out(encoder.feature_names_in_))
    processed_df = pd.DataFrame(np.hstack([numerical_data.values, encoded_data]), columns=all_feature_names)

    # Reorder columns to match training
    processed_df = processed_df.reindex(columns=feature_names, fill_value=0)

    print(f"Processed input data shape: {processed_df.shape}")
    return processed_df

def prediction(input_data):
    """
    Make a prediction using the Random Forest model.
    
    Parameters:
    input_data (dict): Dictionary containing input features
    
    Returns:
    str: "High" or "Low" yield prediction
    """
    # Create DataFrame from input data
    features = pd.DataFrame({
        'Crop_Year': [input_data['Crop_Year']],
        'Annual_Rainfall': [input_data['Annual_Rainfall']],
        'Pesticide': [input_data['Pesticide']],
        'Fertilizer': [input_data['Fertilizer']],
        'Production': [input_data['Production']],
        'Area': [input_data['Area']],
        'Crop': [input_data['Crop']],
        'Season': [input_data['Season']],
        'State': [input_data['State']]
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transformed_features = preprocess_input_data(features, encoder, feature_names)
        predicted_class = model.predict(transformed_features)

    return "High" if predicted_class[0] == 1 else "Low"

def make_batch_predictions(input_array):
    """
    Make predictions for multiple inputs.
    
    Parameters:
    input_array (list): List of dictionaries, each containing input features
    
    Returns:
    list: List of predictions ("High" or "Low")
    """
    results = []
    for input_data in input_array:
        try:
            result = prediction(input_data)
            results.append({
                'input': input_data,
                'prediction': result
            })
        except Exception as e:
            results.append({
                'input': input_data,
                'error': str(e)
            })
    return results

# Example usage with array input
if _name_ == "_main_":
    # Example array of inputs
    input_array = [
        {
            'Crop_Year': 2018,
            'Annual_Rainfall': 1200.5,
            'Pesticide': 100.0,
            'Fertilizer': 200.0,
            'Production': 1500.0,
            'Area': 300.0,
            'Crop': 'Rice',
            'Season': 'Kharif',
            'State': 'Karnataka'
        },
        {
            'Crop_Year': 2019,
            'Annual_Rainfall': 800.0,
            'Pesticide': 80.0,
            'Fertilizer': 150.0,
            'Production': 1200.0,
            'Area': 250.0,
            'Crop': 'Wheat',
            'Season': 'Rabi',
            'State': 'Punjab'
        },
        {
            'Crop_Year': 2020,
            'Annual_Rainfall': 950.0,
            'Pesticide': 90.0,
            'Fertilizer': 180.0,
            'Production': 1300.0,
            'Area': 280.0,
            'Crop': 'Maize',
            'Season': 'Zaid',
            'State': 'Maharashtra'
        }
    ]
    
    # Make batch predictions
    results = make_batch_predictions(input_array)
    
    # Print results
    print("\nRandom Forest Predictions:")
    for i, result in enumerate(results):
        if 'prediction' in result:
            input_data = result['input']
            print(f"\nInput {i+1}:")
            print(f"  Crop: {input_data['Crop']}")
            print(f"  State: {input_data['State']}")
            print(f"  Season: {input_data['Season']}")
            print(f"  Year: {input_data['Crop_Year']}")
            print(f"  Prediction: {result['prediction']} Yield Expected")
        else:
            print(f"\nInput {i+1}: Error - {result['error']}")
            
    # Example of single prediction
    print("\nSingle prediction example:")
    single_input = input_array[0]
    single_result = prediction(single_input)
    print(f"The predicted yield category for {single_input['Crop']} in {single_input['State']} during {single_input['Season']} in {single_input['Crop_Year']} is: {single_result}")