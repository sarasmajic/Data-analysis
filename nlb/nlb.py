import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("cs-data.csv")

corr_mat = df.corr()

positive_corr = corr_mat
sns.heatmap(positive_corr, annot =True, cmap='Blues', mask = positive_corr.isnull())
plt.show() #vizualizacija matrike

df = df.dropna(axis=0, how='any')
x = df.iloc[:,2:].values
y = df.iloc[:, 1].values

print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
scaler = StandardScaler() 
x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test) 

lin_reg = linear_model.LinearRegression()
lin_reg.fit(x_train, y_train)

lin_reg_pred = lin_reg.predict(x_test)
print("Coefficients:\n", lin_reg.coef_)
print("Intercept:\n", lin_reg.intercept_)

print("Mean squared error: %.2f" % mean_squared_error(y_test, lin_reg_pred))
#print(corr_mat) matrika

mlp = MLPClassifier(hidden_layer_sizes=(8,8), activation="relu", solver="adam")
mlp.fit(x_train, y_train)

predict_train = mlp.predict(x_train)
predict_test = mlp.predict(x_test)

print(confusion_matrix(y_train, predict_train))
print(classification_report(y_train, predict_train))

