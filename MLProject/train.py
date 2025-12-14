import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

# Load dataset
df = pd.read_csv("Housing (1).csv")

# One-hot encoding untuk kolom kategorikal
df = pd.get_dummies(df, drop_first=True)

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

mlflow.sklearn.log_model(model, "model")

print("Training selesai via CI")
