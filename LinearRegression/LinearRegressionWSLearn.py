import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Veri Yükleme
data = pd.read_csv("C:\\Users\\pc\\OneDrive\\Masaüstü\\ders\\LAB\\LinearRegression\\housing.csv")
X = data[['total_rooms', 'median_income']]
y = data['median_house_value']

# 2. Veriyi egitim ve test olarak böl (opsiyonel ama tavsiye edilir)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Modeli oluştur ve eğit
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Tahmin yap
predictions = model.predict(X_test)

# 5. MSE hesapla
mse = mean_squared_error(y_test, predictions)
print("MSE (scikit-learn LinearRegression):", mse)

# 6. Grafik
plt.figure(figsize=(8,5))
plt.scatter(X_test['median_income'], y_test, color='blue', label='Gerçek')
plt.scatter(X_test['median_income'], predictions, color='red', label='Tahmin')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Linear Regression with scikit-learn')
plt.legend()
plt.grid()
plt.show()

# 7. Katsayılar
print("\nKatsayılar (theta):")
print("Bias (intercept):", model.intercept_)
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
