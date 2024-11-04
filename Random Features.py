import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# Load the dataset
df = pd.read_csv('dataset.csv')
df = df.drop(columns=['ID','Week8_Total'])  # Remove 'ID' if it's not useful for prediction
y = df['Grade']  # Target variable

# Function to evaluate random features and visualize results
def evaluate_random_features(num_features):
    # Randomly select 'num_features' columns (excluding 'Grade')
    random_columns = np.random.choice(df.drop(columns=['Grade']).columns, num_features, replace=False)
    X = df[random_columns]  # Use selected columns as features

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Support Vector Regressor': SVR(kernel='rbf'),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    # Predict and collect performance metrics
    predictions = {}
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Store predictions for plotting
        predictions[model_name] = y_pred
        
        # Performance metrics
        r2 = r2_score(y_test, y_pred) 
        mse = mean_squared_error(y_test, y_pred) 
        mae = mean_absolute_error(y_test, y_pred)
        print(f"{model_name} with {num_features} features:")
        print(f" - Selected Columns: {list(random_columns)}")
        print(f" - R² Score: {r2: .5f}")
        print(f" - Mean Squared Error (MSE): {mse: .5f}")
        print(f" - Mean Absolute Error (MAE): {mae:.5f}\n")

    # Plot Actual vs Predicted
    plt.figure(figsize=(14, 10))
    for i, (model_name, y_pred) in enumerate(predictions.items(), 1):
        plt.subplot(2, 3, i)
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.title(f'{model_name}: Actual vs Predicted')
        plt.xlabel('Actual Grades')
        plt.ylabel('Predicted Grades')

    plt.tight_layout()
    plt.show()

    # Feature importance for tree-based models
    for model_name, model in [('Random Forest', models['Random Forest']), ('Gradient Boosting', models['Gradient Boosting'])]:
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            features_df = pd.DataFrame({'Feature': random_columns, 'Importance': feature_importances})
            features_df = features_df.sort_values(by='Importance', ascending=False).head(3)

            print(f"\nTop 3 Important Features for {model_name} (using {num_features} random features):")
            print(features_df)

            # Plot feature importances
            plt.figure(figsize=(10, 6))
            sns.barplot(x=features_df['Importance'], y=features_df['Feature'])
            plt.title(f'Feature Importance from {model_name} with {num_features} Features')
            plt.show()

# Test with a random selection of 5 and 10 features
evaluate_random_features(5)
evaluate_random_features(10)
