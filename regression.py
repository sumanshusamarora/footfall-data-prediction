import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import joblib

data = pd.read_pickle("data/data.pkl")
X = data.drop(columns=["ds", "y", "dayofweek"])
y = np.array(data.y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=.2)

model = Pipeline([
    ("scaler", MinMaxScaler((-1,1))),
    ("reg", LinearRegression()),
    ])
model.fit(X, y)

y_hat = model.predict(X_test)
print(f"R squared - {r2_score(y_true=y_test, y_pred=y_hat)}")

plt.plot(y_test, color="green")
plt.plot(y_hat, color="orange")

joblib.dump(model, 'models/linear_regression.pkl')