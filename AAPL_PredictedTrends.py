import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

TICKER = "AAPL"   
START_DATE = "2022-01-01"
END_DATE = "2024-09-01"


df = yf.download(TICKER, start=START_DATE, end=END_DATE)


df["Return"] = df["Close"].pct_change()
df["MA5"] = df["Close"].rolling(window=5).mean()
df["MA10"] = df["Close"].rolling(window=10).mean()
df["Volatility"] = df["Return"].rolling(window=10).std()


df = df.dropna()
    
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)


features = ["Return", "MA5", "MA10", "Volatility"]
X = df[features]
y = df["Target"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


df["Predicted"] = np.nan
df.loc[X_test.index, "Predicted"] = y_pred

plt.figure(figsize=(12,6))
plt.plot(df.index, df["Close"], label="Close Price")
plt.scatter(
    df.index, df["Close"],
    c=df["Predicted"], cmap="coolwarm", alpha=0.6,
    label="Predicted Trend"
)
plt.title(f"{TICKER} Stock Price with Predicted Trends")
plt.legend()
plt.show()
