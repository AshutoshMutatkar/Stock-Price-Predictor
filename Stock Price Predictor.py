import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# get stock data from yahoo finance
ticker = input("Enter stock ticker(AAPL): ")
data = yf.download(ticker, start="2021-01-01", end="2023-12-31")

# sample data
print("\n Sample Data: ")
print(data.head())

#using open and close to predict price
data= data[['Open','Close']].dropna() # keep only open and close columns
X = data[['Open']] # feature (what we know)
y = data['Close'] # target(what we want to predict)

#Split data into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# build and train linear regression model
model= LinearRegression()
model.fit(X_train,y_train)

# predict on test set
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test,y_pred)
print(f"\n Mean Squared Error: {mse:.2f}")

# Visualize actual vs predicted close
plt.scatter(X_test, y_test, color='blue', label='Actual Close Price')
plt.scatter(X_test, y_test, color='red', label='Predicted Close Price')
plt.xlabel('Open Price')
plt.ylabel('Close Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.legend()
plt.show