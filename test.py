# test.py

from pbo.data import YahooAsset
from pbo.strategies import MACrossoverFamily
from pbo.selection import TopOneSelection, TopKSelection
from pbo.api import estimate_overfitting_risk


def main():

    # -------------------------------
    # Load asset
    # -------------------------------
    asset = YahooAsset(
        ticker="NIFTYBEES.NS",
        start="2012-01-01",
        end=None
    )

    # -------------------------------
    # Strategy family (non-degenerate)
    # -------------------------------
    param_grid = [
        {"short": s, "long": l}
        for s in [5, 10, 20, 30]
        for l in [50, 100, 150, 200]
        if s < l
    ]

    strategy_family = MACrossoverFamily(param_grid)

    print(f"Number of strategies: {len(param_grid)}")

    # -------------------------------
    # Top-1 selection
    # -------------------------------
    print("\nRunning Top-1 selection...")

    report_top1 = estimate_overfitting_risk(
        asset=asset,
        strategy_family=strategy_family,
        selection_rule=TopOneSelection(),
        cost=0.001,
        S=8,
        k=4
    )

    print(report_top1.summary())

    # -------------------------------
    # Top-10% selection
    # -------------------------------
    print("\nRunning Top-10% selection...")

    report_top10 = estimate_overfitting_risk(
        asset=asset,
        strategy_family=strategy_family,
        selection_rule=TopKSelection(0.10),
        cost=0.001,
        S=8,
        k=4
    )

    print(report_top10.summary())

    # -------------------------------
    # Sanity checks
    # -------------------------------
    assert 0.0 <= report_top1.pbo <= 1.0
    assert 0.0 <= report_top10.pbo <= 1.0
    assert len(report_top1.lambdas) > 0
    assert len(report_top10.lambdas) > 0

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
