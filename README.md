# MLB Game Totals Model

Predict over/under totals for MLB games and find +EV betting opportunities.

## Setup

```bash
cd "mlb-totals"
pip install -r requirements.txt
```

## Build Order

1. `python src/fetch_data.py`      — pull game results + starters (2018-2024) via MLB Stats API
2. `python src/fetch_stats.py`     — pull pitcher + team batting stats via pybaseball/FanGraphs
3. Download SBR historical lines   — see data/raw/README_lines.md
4. `python src/build_features.py`  — join everything into one feature matrix
5. `python src/train.py`           — train Ridge regression baseline
6. `python src/backtest.py`        — walk-forward backtest + CLV evaluation

## Data Sources

| Source | Data | Cost |
|--------|------|------|
| MLB Stats API (`statsapi`) | Game results, lineups, starters | Free |
| FanGraphs via `pybaseball` | Pitcher FIP/xFIP, team wRC+ | Free |
| Sports Book Reviews Online (SBR) | Historical totals lines | Free |
| Open-Meteo | Weather (temp, wind) | Free |

## Key Features

- Starting pitcher FIP, xFIP, K/9, BB/9
- Bullpen ERA/FIP (rolling 14-day)
- Team wRC+ (rolling 14-day)
- Park runs factor
- Temperature + wind speed/direction
- Rest days, days since SP last start

## Evaluating Edge

Primary metric: **Closing Line Value (CLV)** — consistently beating the closing line
is the strongest signal of real edge, more predictive than raw ROI over small samples.
