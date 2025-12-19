# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1- Import necessary libraries 2- Load the dataset 3- Explore and preprocess the data 4- Define feature matrix X and target vector y 5- Initialize the Decision Tree Regressor model 6- Train the model using fit() 7- Make predictions using predict() 8- Visualize the model results (optional) 9- Evaluate model performance (optional) 10- Tune hyperparameters if needed
   
## Program:
```
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Step 1: Create the dataset
data = pd.DataFrame({
    'YearsExperience': [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.7, 3.9,
                        4.0, 4.5, 4.9, 5.1, 5.3, 5.9, 6.0, 6.8, 7.1, 7.9],
    'Salary': [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 57189, 63218,
               55794, 56957, 57081, 61111, 66029, 83088, 81363, 93940, 91738, 98273]
})

# Step 2: Define features and target
X = data[['YearsExperience']]  # 2D array for sklearn
y = data['Salary']             # 1D array

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the Decision Tree Regressor
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)

# Step 5: Predict on test data
y_pred = regressor.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 7: Visualize the Decision Tree Regression results (without warning)
X_grid = pd.DataFrame(np.arange(min(X['YearsExperience']), max(X['YearsExperience']), 0.01), columns=['YearsExperience'])

plt.scatter(X, y, color='red', label='Actual Salary')
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Model Prediction')
plt.title('Decision Tree Regression: Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Harini SK
RegisterNumber:  25018849
*/
```

## Output:

<img width="1047" height="645" alt="Screenshot 2025-12-19 125353" src="https://github.com/user-attachments/assets/659c02bf-40ab-4356-8670-914036f99212" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
