# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize Dataset and Parameters
2. Apply Sigmoid Function
3. Update Parameters using Gradient Descent
4. Predict Output Class

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: ISHWARYA M
RegisterNumber: 212225230107
*/
# Implementation of Logistic Regression using Gradient Descent
# Fixed Prediction Version

import numpy as np
import matplotlib.pyplot as plt

# Input Data
X = np.array([1, 2, 3, 4, 5, 6])
y = np.array([0, 0, 0, 1, 1, 1])

# Parameters
m = len(X)
w = 0.0
b = 0.0
alpha = 0.01
iterations = 5000

# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient Descent
for i in range(iterations):
    z = w * X + b
    y_pred = sigmoid(z)

    dw = (1/m) * np.sum((y_pred - y) * X)
    db = (1/m) * np.sum(y_pred - y)

    w = w - alpha * dw
    b = b - alpha * db

# Final Prediction on Training Data
final_pred = sigmoid(w * X + b)

print("Weight =", w)
print("Bias =", b)

# Graph
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, final_pred, color='red', linewidth=2, label='Logistic Curve')

plt.title("Logistic Regression using Gradient Descent")
plt.xlabel("Input")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.show()

# User Input
x_new = float(input("Enter input value: "))
prob = sigmoid(w * x_new + b)

print("Probability =", prob)

if prob >= 0.5:
    print("Predicted Class: 1")
else:
    print("Predicted Class: 0")
```

## Output:
![alt text](<Screenshot (270).png>)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

