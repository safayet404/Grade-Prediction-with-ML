# Step 1: Data Processing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset.csv')

# Remove 'ID' as it is not useful for prediction
df = df.drop(columns=['ID','Week8_Total'])

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Our target is the 'Grade' column, everything else is a feature
X = df.drop(columns=['Grade'])  # Features
y = df['Grade']  # Target (Grade)

# Step 2: Split the data into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Model Training with additional models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Model 2: Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Model 3: Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_scaled, y_train)

# Model 4: Support Vector Regressor
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_scaled, y_train)

# Model 5: Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_train)

# Step 5: Performance Evaluation
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Predict using all models
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_dt = dt_model.predict(X_test_scaled)
y_pred_svr = svr_model.predict(X_test_scaled)
y_pred_gb = gb_model.predict(X_test_scaled)

# Calculate R², MSE, MAE for each model
models = {
    'Linear Regression': y_pred_lr,
    'Random Forest': y_pred_rf,
    'Decision Tree': y_pred_dt,
    'Support Vector Regressor': y_pred_svr,
    'Gradient Boosting': y_pred_gb
}

for model_name, y_pred in models.items():
    r2 = r2_score(y_test, y_pred) 
    mse = mean_squared_error(y_test, y_pred) 
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{model_name}:")
    print(f" - R² Score: {r2 : .5f}%")
    print(f" - Mean Squared Error (MSE): {mse : .5f}%")
    print(f" - Mean Absolute Error (MAE): {mae : .5f}%")
    print()

# Step 6: Plot Actual vs Predicted for all models
plt.figure(figsize=(14, 10))

# Linear Regression
plt.subplot(2, 3, 1)
plt.scatter(y_test, y_pred_lr)
plt.title('Linear Regression: Actual vs Predicted')
plt.xlabel('Actual Grades')
plt.ylabel('Predicted Grades')

# Random Forest
plt.subplot(2, 3, 2)
plt.scatter(y_test, y_pred_rf)
plt.title('Random Forest: Actual vs Predicted')
plt.xlabel('Actual Grades')
plt.ylabel('Predicted Grades')

# Decision Tree
plt.subplot(2, 3, 3)
plt.scatter(y_test, y_pred_dt)
plt.title('Decision Tree: Actual vs Predicted')
plt.xlabel('Actual Grades')
plt.ylabel('Predicted Grades')

# Support Vector Regressor
plt.subplot(2, 3, 4)
plt.scatter(y_test, y_pred_svr)
plt.title('SVR: Actual vs Predicted')
plt.xlabel('Actual Grades')
plt.ylabel('Predicted Grades')

# Gradient Boosting
plt.subplot(2, 3, 5)
plt.scatter(y_test, y_pred_gb)
plt.title('Gradient Boosting: Actual vs Predicted')
plt.xlabel('Actual Grades')
plt.ylabel('Predicted Grades')

# Show the plots
plt.tight_layout()
plt.show()

# Step 7: Feature Importance for Tree-Based Models
for model_name, model in zip(['Random Forest', 'Gradient Boosting'], [rf_model, gb_model]):
    feature_importances = model.feature_importances_
    features_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    features_df = features_df.sort_values(by='Importance', ascending=False)
    
    print(f"\nTop 3 Important Features for {model_name}:")
    print(features_df.head(3))
    
    # Visualize feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x=features_df['Importance'], y=features_df['Feature'])
    plt.title(f'Feature Importance from {model_name}')
    plt.show()
