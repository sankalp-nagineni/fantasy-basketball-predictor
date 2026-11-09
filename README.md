# Fantasy Basketball Predictor

A machine learning pipeline that predicts per-game fantasy basketball points for NBA players, then aggregates those predictions into draft rankings and a current-season form board for my Yahoo head-to-head points league.

## Motivation

I play fantasy basketball in a Yahoo H2H points league, and I wanted to find out whether a trained model could rank players more accurately than the rolling-average heuristic most managers (including me) rely on. I am a tenth grader preparing for a future data analyst internship, and I wrote this project to work through a complete ML lifecycle on my own data rather than follow another tutorial. The constraint I set at the start was that every modeling decision had to be defensible in an interview, not just functional in a notebook.

## Results

| Metric | Baseline (10-game rolling average) | LightGBM |
|---|---|---|
| Per-game MAE | 10.19 FP | **10.09 FP** |
| Season rank correlation (Spearman) | **0.994** | 0.990 |
| R² | — | 0.426 |

LightGBM improved on the baseline by roughly one percent of MAE, but the baseline produced slightly more accurate full-season rankings. I treat that as the most interesting finding of the project. When predictions are summed across eighty-plus games, per-game errors cancel out and both methods converge to nearly identical season totals. Model complexity stops earning its keep somewhere around the point where the law of large numbers takes over.

## Pipeline

Five Jupyter notebooks, each a self-contained stage with inputs, outputs, and a written insight statement.

1. `module_1_data_audit.ipynb` — four seasons of player game logs pulled from `nba_api`, audited across scope, integrity, column quality, value sanity, and time structure. Final dataset: 104,641 rows, 925 players.
2. `module_2_feature_engineering.ipynb` — nine leakage-safe features: rest days (capped and decomposed into a return-from-absence flag), rolling fantasy points over 5 and 10 games, rolling minutes, rolling standard deviation, momentum (short minus long rolling mean), home/away, and opponent fantasy points allowed over the previous 10 games.
3. `module_3_modeling.ipynb` — LightGBM trained on 2021–2024 and tested on 2024–25 using a chronological split. Compared directly against the rolling-average baseline on MAE, RMSE, and R².
4. `module_4_projections.ipynb` — per-game predictions aggregated into player-season projections. Two rankers produced: a season total (sum) and a per-game average filtered to a thirty-game minimum. Both compared against baseline using Spearman rank correlation.
5. `module_5_predict_current_season.ipynb` — model re-trained on five full seasons (2021–2026) and used to predict the in-progress 2026–27 season. Outputs a current-form ranker that refreshes on live data.

## Key technical decisions

**Chronological train/test split.** I never randomly shuffled the data. A random split would allow the model to train on future games to predict past ones, which produces inflated test metrics while hiding a model that will fail the moment it reaches production.

**Leakage-safe rolling features.** Every rolling feature uses `.shift(1)` before `.rolling(N)` so that the current game is excluded from its own window. I caught an early bug in my rolling calculation by sanity-checking a specific player's first six games against expected values; the groupby context had been lost between `.shift` and `.rolling`, which rolled values across player boundaries. The fix was moving the entire chain inside a single `transform`.

**Rolling-average baseline.** Every regression metric is reported alongside the baseline. A model that cannot beat a rule computable in Excel does not earn its place in production, and reporting ML metrics in isolation is the single easiest way to look more useful than I actually am.

**Two rankers, two use cases.** A full-season draft board rewards durability and uses the season-total projection. Mid-season decisions (waiver claims, weekly matchups, DFS lineups) should use a per-game average with a sample-size floor. Collapsing the two into a single ranking would conflate two different questions.

**Ablation over feature importance.** I added a rolling standard deviation feature that immediately became a top-three feature by LightGBM importance, but MAE barely moved when I removed it. Importance measures how often a tree splits on a feature, not whether that feature carries new information beyond existing correlated features. Ablation is the only honest test.

## What I learned

- Target leakage is the worst failure mode in ML because it does not show up in test metrics. It only appears after deployment, when the leaked future information no longer exists.
- Feature engineering accounts for most of the gain. The difference between my baseline and my model is small compared to the difference that features of this shape made over raw columns.
- Rank correlation and MAE answer different questions. Reporting only one makes you blind to the other.
- Debugging AI-generated code is itself a core skill, especially for anyone working with pandas groupby chains where intermediate state is easy to lose.

## Limitations

- Only one test season (2024–25). A stronger evaluation would cross-validate across multiple holdout seasons.
- Box-score features only. Injury reports, Vegas totals, and starting-lineup announcements are the most likely sources of further improvement; adding more variations of rolling averages shows clear diminishing returns.
- Tree models regress to the mean. The model systematically under-predicts elite players (Jokić, Luka, Wembanyama) and over-predicts replacement-level players. For the top five picks of a draft, I would trust the baseline over LightGBM.
- No hyperparameter tuning. The model uses sensible defaults (500 estimators, learning rate 0.05, 31 leaves) because at a one-percent gain over baseline, further tuning is not the right next investment.

## Reproducing

All data is pulled live from `nba_api`; nothing is committed. Dependencies in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in order (`module_1` through `module_5`). Module 5 re-trains and refreshes against the live NBA API; the top-30 current-form ranker at the bottom of that notebook is the actionable output I use during the season.
