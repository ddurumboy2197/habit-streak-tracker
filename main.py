import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# O'zgaruvchilar ro'yxati
columns = [
    "DepTime", "ArrTime", "ActualElapsedTime", "AirTime", "ArrDelay", "DepDelay", "Distance"
]

# Ma'lumotlar manbasi
data = {
    "DepTime": [700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600],
    "ArrTime": [900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800],
    "ActualElapsedTime": [120, 120, 120, 120, 120, 120, 120, 120, 120, 120],
    "AirTime": [120, 120, 120, 120, 120, 120, 120, 120, 120, 120],
    "ArrDelay": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "DepDelay": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Distance": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
}

# Pandas datasi yaratish
df = pd.DataFrame(data, columns=columns)

# O'zgaruvchilarni o'rganish
X = df[["DepTime", "ArrTime", "ActualElapsedTime", "AirTime", "Distance"]]
y = df["ArrDelay"]

# Test va train qismlari yaratish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model yaratish
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Modelni mashq qilish
model.fit(X_train, y_train)

# Modelni baholash
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Modelni ishlatish
def predict_arr_delay(dep_time, arr_time, actual_elapsed_time, air_time, distance):
    input_data = pd.DataFrame({
        "DepTime": [dep_time],
        "ArrTime": [arr_time],
        "ActualElapsedTime": [actual_elapsed_time],
        "AirTime": [air_time],
        "Distance": [distance]
    })
    return model.predict(input_data)[0]

# Misol ishlatish
print(predict_arr_delay(700, 900, 120, 120, 100))
```

Kodda quyidagilar mavjud:

*   O'zgaruvchilar ro'yxati yaratilgan.
*   Ma'lumotlar manbasi yaratilgan.
*   Pandas datasi yaratilgan.
*   O'zgaruvchilar o'rganilgan.
*   Test va train qismlari yaratilgan.
*   Model yaratilgan.
*   Modelni mashq qilish.
*   Modelni baholash.
*   Modelni ishlatish uchun funksiya yaratilgan.
*   Misol ishlatish.
