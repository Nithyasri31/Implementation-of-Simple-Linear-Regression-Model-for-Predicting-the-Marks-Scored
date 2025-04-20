# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the needed packages.
2. Assigning hours to x and scores to y.
3.  Plot the scatter plot.
4. se mse,rmse,mae formula to find the values
 

## Program:
```

Developed by: Nithyasri M
RegisterNumber: 212224040226
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

![image](https://github.com/user-attachments/assets/690aef75-d19f-411b-b812-75801cd36670)
![image](https://github.com/user-attachments/assets/d7891398-2a52-453d-82f6-99042693c831)
![image](https://github.com/user-attachments/assets/301f58c3-d919-4855-8943-119068f3439c)
![image](https://github.com/user-attachments/assets/4f94b434-0670-43fa-a40f-ded025ee088e)
![image](https://github.com/user-attachments/assets/0c748e45-406d-4fe5-a3e7-2cc072f61b16)
![image](https://github.com/user-attachments/assets/f30f4eae-a207-4ed9-a126-d2715b9ef739)
![image](https://github.com/user-attachments/assets/d628ecd5-9dc5-447f-a5d9-dc09d285e6ab)
![image](https://github.com/user-attachments/assets/40dd07ca-c64f-4f8f-8603-2dd5177734e8)
![image](https://github.com/user-attachments/assets/de93b4e3-7c83-43da-9a2d-4800abe87eea)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
