import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def multiple_models(dataset_path):
    # Load the data
    df = pd.read_csv(dataset_path)
    
    # Drop unnecessary columns
   # df.drop(['Unnamed: 0'], axis=1, inplace=True)
    
    # Update column names based on your dataset
    df.columns = ['day', 'month', 'year', 'Temperature', ' RH', ' Ws', 'Rain ', 'FFMC', 
                  'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Classes  ']
    
    # Extract object columns for encoding
    object_cols = df.select_dtypes(include='object').columns
    # Step 1: Clean the column by stripping spaces and converting to lowercase
    df['Classes  '] = df['Classes  '].str.strip().str.lower()

    df = df[df['Classes  '] != 'classes']

    df = df.dropna(subset=['Classes  '])

    unique_values = df['Classes  '].unique()
    print("Unique values before replacement:", unique_values)    
    # Encode categorical columns (if there are any)
    mappings = {}
    for col in object_cols:
        unique_values = df[col].unique()
        mapping = {value: idx for idx, value in enumerate(unique_values)}
        mappings[col] = mapping
        df[col] = df[col].map(mapping)
    
    # Define input and output columns
    ind_col = [col for col in df.columns if col != "Classes  "]  # Independent variables
    dep_col = "Classes  "  # Dependent variable (target)
    
    # Split the data into features (X) and target (y)
    X = df[ind_col]
    y = df[dep_col]
    
    # Split the data into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    
    # Initialize models
    rf_model = RandomForestClassifier(random_state=0)
    xgb_model = XGBClassifier(random_state=0)
    lr_model = LogisticRegression(max_iter=1000)  # Logistic Regression with higher iterations for convergence
    
    # Train the models
    rf_model.fit(X_train, y_train)
    print("Random Forest classifier trained")
    
    xgb_model.fit(X_train, y_train)
    print("XGBoost classifier trained")
    
    lr_model.fit(X_train, y_train)
    print("Logistic Regression classifier trained")
    
    # Make predictions for each model
    rf_test_pred = rf_model.predict(X_test)
    xgb_test_pred = xgb_model.predict(X_test)
    lr_test_pred = lr_model.predict(X_test)
    
    # Function to return metrics
    def get_metrics(y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),  # Convert to list for JSON serializable format
            'classification_report': classification_report(y_true, y_pred, output_dict=True)  # Dictionary format for report
        }
    
    # Store results for each model
    results = {
        'random_forest': {
            'test': get_metrics(y_test, rf_test_pred)
        },
        'xgboost': {
            'test': get_metrics(y_test, xgb_test_pred)
        },
        'logistic_regression': {
            'test': get_metrics(y_test, lr_test_pred)
        }
    }
    
    return results

# Example usage:
# results = multiple_classification_models('path_to_your_dataset.csv')
# print(results)
