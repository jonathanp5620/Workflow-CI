import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

df = pd.read_csv("Housing (1).csv")

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("Housing_CI_Experiment")

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mlflow.log_metric("MAE", mean_absolute_error(y_test, y_pred))
    mlflow.log_metric("MSE", mean_squared_error(y_test, y_pred))
    mlflow.log_metric("R2", r2_score(y_test, y_pred))

    mlflow.sklearn.log_model(model, "model")

print("Training selesai via CI")
