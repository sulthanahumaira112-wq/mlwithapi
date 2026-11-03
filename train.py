import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import r2_score
X = np.array([
    [500],
    [700],
    [800],
    [1000],
    [1200],
    [1500],
    [1800],
    [2000]
])

y = np.array([
    150000,
    200000,
    220000,
    280000,
    320000,
    400000,
    470000,
    520000
])


X_train,X_test,y_train,y_test=train_test_split(
     X, y, test_size=0.25, random_state=42
)

model=LinearRegression()
model.fit(X_train,y_train)
pred=model.predict(X_test)
print("R2 Score:", r2_score(y_test, pred))

joblib.dump(model,"house_model.pkl")
print("Model saved")

