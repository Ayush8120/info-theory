"""
This file uses the methods from logistic_regression.py to illustrate simple concepts from the 7480 information theory class
First version: 10/28/2024
This version: 11/12/2024
https://northeastern-datalab.github.io/cs7840/fa24/calendar.html
"""


import numpy as np
from sklearn.linear_model import LogisticRegression
from logistic_regression import(create_matrix_plot)


# Create input binary indicator (n x m) matrix X, and n-dimensional label vector y with k different labels
CHOICE = 5

filetype="png"
C = 1e5         # regularization for logistic regression. Default is 1. The higher, the less regularization
j = 2           # column (attribute) in training set
c = 1           # column in labels and prediction sets

if CHOICE == 1:
    X = np.array([[0, 1, 0, 1],
                  [1, 0, 1, 0],
                  [0, 0, 1, 1],
                  [1, 1, 0, 0],
                  [0, 1, 1, 0]])
    n = 5
    m = 4

    y = np.array([0, 1, 2, 0, 1])   # Multinomial classes
    k = 3

elif CHOICE == 2:
    X = np.array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 0, 1],
                  [1, 1, 0],
                  [0, 1, 1]])
    n = 5
    m = 3

    y = np.array([0, 1, 0, 0, 1])  # Multinomial classes
    k = 2

elif CHOICE == 3:
    X = np.array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 0, 1],
                  [1, 1, 0],
                  [0, 1, 1],
                  [0, 1, 1],
                  [1, 1, 1]])
    n = 6
    m = 3

    y = np.array([0, 1, 1, 0, 1, 0, 0])  # Multinomial classes
    k = 2
    # C = 1e5


elif CHOICE == 4:   # tennis
    X = np.array([[1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
                  [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                  [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                  [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 0, 1, 0, 0, 1, 0, 1, 1, 0],
                  [0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
                  [0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
                  [1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
                  [1, 0, 0, 0, 0, 1, 0, 1, 1, 0],
                  [0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
                  [1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                  [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
                  [0, 1, 0, 1, 0, 0, 0, 1, 1, 0],
                  [0, 0, 1, 0, 1, 0, 1, 0, 0, 1]])
    n = 14
    m = 10

    y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0])  # Multinomial classes
    k = 2
    C = 1e5


elif CHOICE == 5:
    X = np.array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 0, 1],
                  [1, 1, 0],
                  [0, 1, 1],
                  [0, 1, 1],
                  [1, 1, 1],
                  [1, 1, 0]])
    n = 7
    m = 3

    y = np.array([0, 1, 1, 0, 1, 0, 0, 1])  # Multinomial classes
    k = 2
    # C = 1e5



# Fit multinomial logistic regression model
model = LogisticRegression(solver='lbfgs', C=C)  # C is regularization. Is 1 by default.
model.fit(X, y)
proba = model.predict_proba(X)  # Predict the probability distribution for each data point

coefficients = model.coef_      # Get the coefficients (parameters for each feature)
intercept = model.intercept_    # Get the intercept (bias term)

print("Coefficients:", coefficients)
print("Intercept:", intercept)


# Represent the original training labels as an indicator matrix
label_indicator_matrix = np.zeros((y.shape[0], k))  # Create a nxk matrix of zeros
label_indicator_matrix[np.arange(y.shape[0]), y] = 1  # Use advanced indexing to mark the true labels


# Compare the weighted column sums from original training labels against the predicted ones
# print(X[:,j])
# print(label_indicator_matrix[:,c])
# print(proba[:,c])
print("j: {}".format(j))
print("c: {}".format(c))
print("Sum of training column:   {}".format(X[:,j].dot(label_indicator_matrix[:,c])))
print("Sum of estimation column: {}".format(X[:,j].dot(proba[:,c])))

# TODO: alternative proof with just the column sums
# print(np.sum(label_indicator_matrix[:,c]))
# print(np.sum(np.sum(proba[:,c])))


# Show Input Data X as Indicator Matrix
create_matrix_plot(X, 'Fig_212_lr_input_data_matrix' + '_' + str(CHOICE), add_text=False, show_colorbar=False,filetype=filetype)

# Show Original Training Labels as Indicator Matrix
create_matrix_plot(label_indicator_matrix, 'Fig_212_lr_label_indicator_matrix' + '_' + str(CHOICE), add_text=False, show_colorbar=False,filetype=filetype)

# Show Predicted Probabilities, with text inside the squares, and color bar
create_matrix_plot(proba, 'Fig_212_lr_predicted_probabilities' + '_' + str(CHOICE), add_text=True, filetype=filetype)


