import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
import pickle

data = pd.read_csv(r"D:\credit score1\credit score\backend\credit_score.csv")

X = data[['INCOME', 'SAVINGS', 'DEBT']]
y = data['CREDIT_SCORE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

predictions = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, predictions))

