# src/turn_prediction/test_live_feature_contract.py

from __future__ import annotations

from live_feature_contract import (
build_live_feature_contract_report,
format_live_feature_contract_report,
)

def main() -> None:
    report = build_live_feature_contract_report()
    print(format_live_feature_contract_report(report))


if __name__ == "__main__":
    main()