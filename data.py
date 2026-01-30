import yfinance as yf

class Asset:
    def load_price(self):
        raise NotImplementedError


class YahooAsset(Asset):
    def __init__(self, ticker, start=None, end=None):
        self.ticker = ticker
        self.start = start
        self.end = end

    def load_price(self):
        data = yf.download(
            self.ticker,
            start=self.start,
            end=self.end,
            progress=False
        )
        return data["Close"].dropna()
