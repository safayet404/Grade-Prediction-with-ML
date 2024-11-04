import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns  # Uncommented this line to use seaborn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Step 1: Data Loading and Initial Exploration
def load_and_explore_data(file_path):
    # Load the data
    df = pd.read_csv('dataset.csv')
    
    # Sample the dataset for initial exploration
    df_sample = df.sample(frac=0.3, random_state=42)  # Adjust the fraction as needed
    
    # Separate features into categories
    assessment_columns = ['Week2_Quiz1', 'Week3_MP1', 'Week3_PR1', 'Week5_MP2', 
                         'Week5_PR2', 'Week7_MP3', 'Week7_PR3', 'Week4_Quiz2', 
                         'Week6_Quiz3']
    
    # Visualize distribution of grades
    plt.figure(figsize=(15, 6))
    df_sample[assessment_columns].boxplot()
    plt.xticks(rotation=45)
    plt.title('Distribution of Assessment Grades (Sample)')
    plt.tight_layout()
    plt.show()
    
    print("Dataset Shape:", df.shape)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nBasic Statistics for Assessments:")
    print(df[assessment_columns].describe())
    
    return df

# Step 2: Data Processing
def process_data(df):
    # Separate features and target
    X = df.drop(['ID', 'Grade', 'Week8_Total'], axis=1)  # Remove ID and Grade (target)
    y = df['Grade']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y

# Step 3: Analyze Activity Patterns
def analyze_activity_patterns(df):
    # Analyze weekly activity patterns
    activity_by_type = {
        'Content Related (Stat0)': [col for col in df.columns if 'Stat0' in col],
        'Assignment Related (Stat1)': [col for col in df.columns if 'Stat1' in col],
        'Grade Related (Stat2)': [col for col in df.columns if 'Stat2' in col],
        'Forum Related (Stat3)': [col for col in df.columns if 'Stat3' in col]
    }
    
    plt.figure(figsize=(15, 8))
    for activity_type, columns in activity_by_type.items():
        weekly_means = df[columns].mean()
        plt.plot(range(1, 10), weekly_means, marker='o', label=activity_type)
    
    plt.xlabel('Week')
    plt.ylabel('Average Activity Count')
    plt.title('Weekly Activity Patterns by Type')
    plt.legend()
    plt.grid(True)
    plt.show()

# Step 4: Train and Evaluate Models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Model 1: Random Forest (with reduced estimators for testing)
    rf_model = RandomForestRegressor(n_estimators=10, random_state=42)  # Reduced for faster testing
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # Model 2: Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    # Calculate performance metrics
    models_performance = {
        'Random Forest': {
            'R2': r2_score(y_test, rf_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred))
        },
        'Linear Regression': {
            'R2': r2_score(y_test, lr_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred))
        }
    }
    
    # Visualize predictions vs actual
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, rf_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Grade')
    plt.ylabel('Predicted Grade')
    plt.title('Random Forest: Predicted vs Actual')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, lr_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Grade')
    plt.ylabel('Predicted Grade')
    plt.title('Linear Regression: Predicted vs Actual')
    
    plt.tight_layout()
    plt.show()
    
    return rf_model, lr_model, models_performance

# Step 5: Feature Importance Analysis
def analyze_feature_importance(rf_model, feature_names):
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    # sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.show()
    
    return feature_importance.head(3)

# Main execution flow
def main():
    # Load and explore data
    print("Step 1: Loading and Exploring Data")
    df = load_and_explore_data('MP2_Data - MP2_Data.csv')  # Replace with your CSV file path
    
    # Process data
    print("\nStep 2: Processing Data")
    X, y = process_data(df)
    
    # Analyze activity patterns
    print("\nStep 3: Analyzing Activity Patterns")
    analyze_activity_patterns(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate models
    print("\nStep 4: Training and Evaluating Models")
    rf_model, lr_model, performance = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    print("\nModel Performance:")
    for model_name, metrics in performance.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
    
    # Analyze feature importance
    print("\nStep 5: Analyzing Feature Importance")
    top_features = analyze_feature_importance(rf_model, X.columns)
    print("\nTop 3 Most Important Features:")
    print(top_features)

if __name__ == "__main__":
    main()