from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import yfinance as yf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- REQUEST MODEL ----------
class StockRequest(BaseModel):
    company: str
    country: Optional[str] = "India"

# ---------- PRICE FETCH ----------
def fetch_prices_and_dates(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="3mo", interval="1d")

    if data.empty:
        return np.array([]), []

    prices = data["Close"].values
    dates = data.index.strftime("%Y-%m-%d").tolist()
    return prices, dates

# ---------- MAIN ANALYSIS ----------
@app.post("/analyze")
def analyze_stock(data: StockRequest):

    symbol = data.company.upper()

    if data.country == "India":
        prices, dates = fetch_prices_and_dates(symbol + ".NS")
        if prices.size == 0:
            prices, dates = fetch_prices_and_dates(symbol + ".BO")
    elif data.country == "UK":
        prices, dates = fetch_prices_and_dates(symbol + ".L")
    else:
        prices, dates = fetch_prices_and_dates(symbol)

    if prices.size == 0:
        return {"status": "error", "message": "Price data not available"}

    # ----- VOLATILITY -----
    returns = np.diff(prices) / prices[:-1]
    volatility = float(np.std(returns))
    # ----- RISK SCORE -----
    risk_score = min(100, round(volatility * 3000, 2))


    # ----- TREND -----
    trend = "UP" if prices[-1] > prices[0] else "DOWN"
    # ----- CONFIDENCE SCORE -----
    confidence_score = max(0, round(100 - risk_score, 2))

    price_change_pct = abs((prices[-1] - prices[0]) / prices[0]) * 100
    if price_change_pct > 5:
        confidence_score = min(100, confidence_score + 10)


    # ----- DECISION LOGIC -----
    LOW_VOL = 0.010
    MID_VOL = 0.020

    if volatility < LOW_VOL:
        decision = "ACT"
        risk_level = "LOW"
        explanation = "Market is stable with low volatility."
    elif volatility < MID_VOL:
        decision = "WAIT"
        risk_level = "MEDIUM"
        explanation = "Moderate volatility. Waiting improves confidence."
    else:
        decision = "ABSTAIN"
        risk_level = "HIGH"
        explanation = "High volatility indicates unstable conditions."

    # ----- CONFIDENCE BAND -----
    mean_price = np.mean(prices)
    band = volatility * mean_price * 2

    upper_band = (prices + band).round(2).tolist()
    lower_band = (prices - band).round(2).tolist()

    return {
        "status": "success",
        "company": data.company,
        "country": data.country,
        "latest_price": round(float(prices[-1]), 2),
        "trend": trend,
        "volatility": round(volatility, 4),
        "decision": decision,
        "risk_level": risk_level,
        "decision_reason": explanation,
        "prices": prices.round(2).tolist(),
        "dates": dates,
        "upper_band": upper_band,
        "lower_band": lower_band,
        "risk_score": risk_score,
        "confidence_score": confidence_score,

    }
