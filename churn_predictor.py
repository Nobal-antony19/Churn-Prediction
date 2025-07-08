import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier # Make sure you have xgboost installed: pip install xgboost
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
from imblearn.over_sampling import SMOTE # Make sure you have imbalanced-learn installed: pip install imbalanced-learn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output, especially from sklearn/imblearn
warnings.filterwarnings('ignore')

def load_data(file_path):
    """
    Loads data from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from: {file_path}")
        print("First 5 rows of the dataset:")
        print(df.head())
        print("\nDataset Information:")
        df.info()
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the path.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def preprocess_data(df, target_column_name):
    """
    Preprocesses the dataframe by separating features and target,
    handling missing values, and setting up transformers for numerical
    and categorical features using a ColumnTransformer.
    """
    if target_column_name not in df.columns:
        print(f"Error: Target column '{target_column_name}' not found in the dataset.")
        return None, None, None

    # Separate features (X) and target (y)
    X = df.drop(columns=[target_column_name])
    y = df[target_column_name]

    # Convert target to binary (0/1) if it's not already
    # This assumes 'Yes'/'1' or similar indicates churn. Adjust as per your data.
    unique_target_values = y.unique()
    if len(unique_target_values) > 2:
        print(f"Warning: Target column '{target_column_name}' has more than two unique values: {unique_target_values}")
        print("This model is designed for binary classification (churn/no churn). Please ensure your target is binary.")
        # Attempt to convert common binary representations
        if 'Yes' in unique_target_values and 'No' in unique_target_values:
            y = y.apply(lambda x: 1 if x == 'Yes' else 0)
        elif '1' in unique_target_values and '0' in unique_target_values:
            y = y.astype(int)
        else:
            print("Could not automatically convert target to binary. Please manually adjust your target column.")
            return None, None, None
    elif len(unique_target_values) == 2:
        # If two unique values, map them to 0 and 1
        val1, val0 = unique_target_values
        # Heuristic: assume the less frequent value is the positive class (churn) if not explicitly 'Yes' or '1'
        if y.value_counts().min() == y.value_counts()[val1]:
             y = y.map({val1: 1, val0: 0})
        else:
             y = y.map({val0: 1, val1: 0})
        print(f"Target column '{target_column_name}' converted to binary: {y.value_counts()}")
    else:
        print(f"Error: Target column '{target_column_name}' has less than two unique values. Cannot perform binary classification.")
        return None, None, None


    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    print(f"\nIdentified Numerical Features: {numerical_features}")
    print(f"Identified Categorical Features: {categorical_features}")

    # Handle missing values (simple imputation for demonstration)
    # For numerical: fill with mean
    for col in numerical_features:
        if X[col].isnull().any():
            X[col].fillna(X[col].mean(), inplace=True)
            print(f"Filled missing values in numerical column '{col}' with mean.")
    # For categorical: fill with mode
    for col in categorical_features:
        if X[col].isnull().any():
            X[col].fillna(X[col].mode()[0], inplace=True)
            print(f"Filled missing values in categorical column '{col}' with mode.")

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns (if any) as they are
    )
    return X, y, preprocessor

def train_and_evaluate_xgboost(X_train, y_train, X_test, y_test, preprocessor):
    """
    Trains and evaluates an XGBoost model.
    Includes SMOTE for imbalanced data and GridSearchCV for hyperparameter tuning.
    """
    print("\n--- Training XGBoost Model ---")

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    print(f"Original training data shape: X={X_train.shape}, y={y_train.shape}, Churn count: {y_train.sum()}")
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Resampled training data shape: X={X_train_resampled.shape}, y={y_train_resampled.shape}, Churn count: {y_train_resampled.sum()}")
    print("SMOTE applied to balance the training dataset.")

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__max_depth': [3, 5, 7],
        'classifier__subsample': [0.7, 0.8, 0.9],
        'classifier__colsample_bytree': [0.7, 0.8, 0.9]
    }

    # Create a pipeline that first preprocesses and then applies the classifier
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    # Perform GridSearchCV for hyperparameter tuning
    print("Starting GridSearchCV for hyperparameter tuning...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_resampled, y_train_resampled)

    best_model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best ROC AUC score during tuning: {grid_search.best_score_:.4f}")

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1] # Probability of churn

    # Evaluate the model
    print("\n--- Model Evaluation on Test Set ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted No Churn', 'Predicted Churn'],
                yticklabels=['Actual No Churn', 'Actual Churn'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for XGBoost Model')
    plt.show()

    # ROC Curve Visualization
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_estimator(best_model, X_test, y_test, name='XGBoost', ax=plt.gca())
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for XGBoost Model')
    plt.legend()
    plt.grid()
    plt.show()

    return best_model

def main():
    """
    Main function to run the churn prediction model workflow using only XGBoost.
    """
    print("--- XGBoost Churn Prediction Model Builder ---")
    print("This script will help you build and evaluate a churn prediction model using XGBoost.")
    print("It includes automated preprocessing, SMOTE for imbalanced data, and hyperparameter tuning.")

    # Changed the file path as requested
    file_path = "D:\\churn_dataset.csv"
    print(f"\nUsing the specified file path: {file_path}")

    df = load_data(file_path)

    if df is None:
        print("\nExiting due to data loading error.")
        return

    print("\n--- Data Preprocessing ---")
    target_column_name = input("Please enter the name of the target/churn column (e.g., 'Churn', 'Exited'): ")

    X, y, preprocessor = preprocess_data(df, target_column_name)

    if X is None or y is None or preprocessor is None:
        print("\nExiting due to preprocessing error.")
        return

    # Split data into training and testing sets
    # Using stratify=y ensures that the proportion of churners is similar in both sets
    print("\nSplitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    print(f"Churn distribution in training set:\n{y_train.value_counts(normalize=True)}")
    print(f"Churn distribution in test set:\n{y_test.value_counts(normalize=True)}")

    # Train and evaluate XGBoost Model
    xgboost_model = train_and_evaluate_xgboost(X_train, y_train, X_test, y_test, preprocessor)

    print("\n--- Model Building Complete ---")
    print("Review the evaluation metrics and plots above to understand the XGBoost model's performance.")
    print("\nImportant Note on 95%+ Accuracy:")
    print("Achieving 95%+ accuracy consistently on diverse, real-world churn datasets is extremely challenging.")
    print("Model performance depends heavily on data quality, feature relevance, and the inherent predictability of churn in your specific dataset.")
    print("If your current model doesn't meet your target, consider:")
    print("1. More advanced Feature Engineering (e.g., creating interaction terms, time-series features).")
    print("2. Collecting more relevant data (e.g., customer sentiment, competitor pricing).")
    print("3. Exploring other advanced models like LightGBM or CatBoost, or even deep learning for very complex patterns.")
    print("4. Focusing on business-critical metrics like Recall (to catch more churners) or Precision (to reduce false positives) based on your strategy.")

if __name__ == "__main__":
    main()
