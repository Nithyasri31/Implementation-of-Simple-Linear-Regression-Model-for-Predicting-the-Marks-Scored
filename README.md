# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the needed packages.
2.Assigning hours to x and scores to y.
3.Plot the scatter plot.
4.Use mse,rmse,mae formula to find the values


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:  
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

df=pd.read_csv('/content/student_scores.csv')

#displaying the content in datafile.

df.head()

df.tail()

X=df.iloc[:,:-1].values

X

Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split

X_train,X_test, Y_train,Y_test=train_test_split(X, Y, test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X_train, Y_train)

Y_pred=regressor.predict(X_test)

Y_pred

Y_test

plt.scatter(X_train, Y_train,color="orange")

plt.plot(X_train, regressor.predict(X_train),color="red")

plt.title("Hours vs Scores (Training Set)")

plt.xlabel("Hours")

plt.ylabel("Scores")

plt.show()

plt.scatter(X_test, Y_test,color="purple")

plt.plot(X_test, regressor.predict(X_test), color="yellow")

plt.title("Hours vs Scores (Test Set)")

plt.xlabel("Hours")

plt.ylabel("Scores")

plt.show()

mse=mean_squared_error(Y_test, Y_pred)

print('MSE = ',mse)

mae=mean_absolute_error(Y_test, Y_pred)

print('MAE = ', mae)

rmse=np.sqrt(mse)

print("RMSE = ",rmse)

```

## Output:
![Screenshot 2025-04-20 124318](https://github.com/user-attachments/assets/644a1024-c50e-45c5-b059-394f9ef9b74e)
![Screenshot 2025-04-20 163002](https://github.com/user-attachments/assets/1c587142-9293-489f-99a0-66238549183d)
![Screenshot 2025-04-20 163014](https://github.com/user-attachments/assets/463d31b4-90aa-4fc0-94b5-e60caa07b3b7)
![Screenshot 2025-04-20 163056](https://github.com/user-attachments/assets/ebafff89-6d4a-4a1c-9e4b-c45e056a8284)
![Screenshot 2025-04-20 163123](https://github.com/user-attachments/assets/bcd7cec1-93f7-4415-bb76-90d9c8250c73)
![Screenshot 2025-04-20 163146](https://github.com/user-attachments/assets/f9eafa36-01c8-4c8d-a836-e8355ad7e13f)
![Screenshot 2025-04-20 163203](https://github.com/user-attachments/assets/4b954932-6062-4495-aa05-967d19eeb91d)
![Screenshot 2025-04-20 163219](https://github.com/user-attachments/assets/c172a78e-e1c0-4907-8fd8-899de69d62dd)
![Screenshot 2025-04-20 163234](https://github.com/user-attachments/assets/57cb1b74-79e8-46c2-a0e2-3d6cbf4f968e)


## Result:


Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
