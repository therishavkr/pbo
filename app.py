# app.py

from fastapi import FastAPI
from pydantic import BaseModel

from pbo.api import estimate_overfitting_risk
from pbo.data import YahooAsset
from pbo.strategies import MACrossoverFamily
from pbo.selection import TopOneSelection, TopKSelection

app = FastAPI(title="PBO Backtest Overfitting API")


# -------------------------------
# Request schema
# -------------------------------
class PBORequest(BaseModel):
    ticker: str
    selection: str = "top1"     # "top1" or "topk"
    top_pct: float = 0.10
    cost: float = 0.001


# -------------------------------
# Health check
# -------------------------------
@app.get("/")
def health():
    return {"status": "ok"}


# -------------------------------
# Main API endpoint
# -------------------------------
@app.post("/estimate_pbo")
def estimate_pbo(req: PBORequest):

    asset = YahooAsset(
        ticker=req.ticker,
        start="2012-01-01",
        end=None
    )

    # Strategy family (MA crossover)
    param_grid = [
        {"short": s, "long": l}
        for s in [5, 10, 20, 30]
        for l in [50, 100, 150, 200]
        if s < l
    ]

    strategy_family = MACrossoverFamily(param_grid)

    # Selection rule
    if req.selection == "topk":
        selection_rule = TopKSelection(req.top_pct)
    else:
        selection_rule = TopOneSelection()

    report = estimate_overfitting_risk(
        asset=asset,
        strategy_family=strategy_family,
        selection_rule=selection_rule,
        cost=req.cost,
        S=8,
        k=4
    )

    return report.summary()
