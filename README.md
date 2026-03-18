# MinuteModel V1: Draft-Only LoL Match Duration Regression

MinuteModel V1 predicts professional League of Legends match duration **immediately after champion draft**, using Oracle's Elixir-style CSV input.

This is a **draft-only benchmark model** designed to be:
- leakage-safe
- chronological (time-aware) in validation
- modular and extensible for future in-game telemetry models

## What This Project Builds

- Schema inspection for raw Oracle's Elixir row structure
- Safe flattening from multi-row raw data to **one row per match (`gameid`)**
- Leakage-safe historical team rolling priors (duration and optional CKPM)
- Extended leakage-safe rolling priors (win/objective/gold-diff/side priors)
- Draft feature generation:
  - role-aware champions (when available)
  - bag-of-champions fallback (configurable)
  - optional champion scaling / composition pace priors (configurable)
  - optional dense draft summary scores (early/scaling/engage/peel/poke/etc.)
  - optional draft interaction features (tempo/snowball-vs-stall style)
  - optional draft-conditional team behavior priors
- Baseline models:
  - global mean
  - league mean
  - league + patch mean
  - ridge regression
- Primary model (default): CatBoost regressor
- Optional primary model: LightGBM regressor
- Strict chronological split (train/validation/test by date order)
- Evaluation outputs (MAE/RMSE/MedAE + within-2/5 minutes)
- Residual and error breakdown plots
- SHAP diagnostics (global + local)
- Inference API for single pre-game input

## Project Structure

- `minutemodel/data_loading.py`
- `minutemodel/schema_inspection.py`
- `minutemodel/preprocessing.py`
- `minutemodel/feature_engineering.py`
- `minutemodel/champion_scaling.py`
- `minutemodel/baselines.py`
- `minutemodel/train.py`
- `minutemodel/evaluate.py`
- `minutemodel/inference.py`
- `minutemodel/config.py`
- `minutemodel/utils.py`
- `minutemodel/main.py`
- `main.py` (entrypoint wrapper)
- `config/example_config.yaml`

## Installation

```bash
python -m pip install -r requirements.txt
```

### Streamlit Community Cloud note

This repo includes `runtime.txt` with `python-3.11` so cloud deploys use a wheel-friendly Python version for
`scikit-learn`/`scipy` (avoids source builds that require a Fortran toolchain).
If Cloud still provisions Python 3.14 for an existing app, this repo also uses `scikit-learn>=1.8.0,<1.9`,
which has Python 3.14 Linux wheels.

## Data Assumptions and Schema Handling

Oracle's Elixir files often contain multiple rows per match (player and/or team rows). This project inspects and handles that explicitly.

### Schema inspection diagnostics

`inspect` reports:
- row counts by `gameid`
- row distributions by `participantid`, `side`, and `position`
- whether team rows exist (for example `position=team`)
- draft field duplication rates across row types
- `gamelength` unit guess (seconds vs minutes)

### Match-level flattening

Flattening uses `gameid` and side normalization (`Blue`/`Red`) to construct one row per game with fields such as:
- `blue_team_id`, `red_team_id`
- `blue_team_name`, `red_team_name`
- side draft champions (role-aware if recoverable)
- side bans/pick slots
- `target_gamelength_seconds`
- `target_gamelength_minutes`

If role recovery is incomplete, the pipeline falls back to bag-of-champions features from picks/champion rows.

Assumptions used when schema is ambiguous:
- side is normalized to `Blue`/`Red` from common tokens (`blue/red`, `100/200`, etc.)
- repeated draft values across player/team rows are resolved by stable mode-first selection
- team identity uses `teamid` when available, otherwise `teamname`
- only matches with both sides present are retained in V1

## Leakage Policy

Version 1 only uses features known at draft completion.

### Allowed feature families (V1)
- pre-game metadata (`league`, `year`, `split`, `playoffs`, `date`, `patch`, side/team identity)
- draft fields (`champion`, `pick1`-`pick5`, `ban1`-`ban5`, role champs)
- strictly historical rolling priors computed from **previous** matches only

### Forbidden feature families (V1)
No current-match in-game/post-game outcomes, including:
- result and combat stats (`kills`, `deaths`, `assists`, etc.)
- objectives (`dragons`, `barons`, `towers`, etc.)
- gold/xp/cs progression or snapshot features (including `@10/@15/@20/@25` and `*diffat*` fields)

The code enforces forbidden feature checks before modeling.
Concrete lists are defined in `minutemodel/config.py` as `ALLOWED_COLUMNS`, `FORBIDDEN_COLUMNS`, and `FORBIDDEN_COLUMN_PATTERNS`.

## Rolling Priors (Leakage-Safe)

For each team-side row in chronological order:
- rolling 10-game average duration (shifted by 1 match)
- optional rolling 10-game CKPM (shifted by 1 match)
- optional rolling win/objective/gold-diff priors (shifted by 1 match)
- optional side-specific win/duration priors (shifted by 1 match)

Additional leakage-safe conditional priors (optional):
- team average duration on early-game archetypes
- team average duration on scaling archetypes
- first-dragon rate on early archetypes
- herald rate vs scaling opponents
- CKPM on skirmish-heavy drafts
- duration in similar archetype-vs-archetype matchups

All of the above are computed with strict `shift(1)` logic so current/future rows cannot leak into current-match features.

Fallback chain when history is short:
1. league expanding prior (historical only)
2. global expanding prior (historical only)
3. dataset mean fallback if still missing

## Champion Scaling / Composition Pace Features (Optional)

When `use_champion_scaling_features: true`, the pipeline adds engineered draft priors based on champion tendencies in historical training data.

How coefficients are built:
- fit on **training split only** (never on validation/test before scoring)
- per champion: `avg_game_length_seconds_when_picked - global_train_avg_seconds`
- optional smoothing shrinkage for low-sample champions (`champion_scaling_min_samples`)
- optional patch-aware coefficients (fall back to champion-level prior when patch sample is small)
- optional recency weighting (half-life in days)

Leakage rule:
- the lookup is fit only inside training (`DraftFeatureBuilder.fit(train_df)`)
- validation/test/inference rows only receive mapped values from that fitted lookup
- unseen champions fall back to `0.0` (global prior delta)

Aggregated team features:
- `blue_scaling_sum`, `red_scaling_sum`
- `blue_scaling_mean`, `red_scaling_mean`
- `blue_scaling_max`, `red_scaling_max`
- `blue_scaling_min`, `red_scaling_min`
- `blue_scaling_known_count`, `red_scaling_known_count`
- `scaling_diff` (blue minus red)

Interpretation:
- these are handcrafted priors, not causal truth
- they complement (do not replace) raw champion identity features

## Configuration

Use YAML config (see `config/example_config.yaml`).
`input_csv` can be a single CSV path or a glob pattern (for example yearly files like `data/*_LoL_esports_match_data_from_OraclesElixir.csv`).

Required/important options:
- `primary_model` (`catboost` or `lightgbm`)
- `use_role_specific_draft_features`
- `use_bag_of_champions_fallback`
- `target_unit` (`seconds` or `minutes`)
- `rolling_window_size`
- `use_champion_scaling_features`
- `use_extended_rolling_team_priors`
- `use_draft_summary_features`
- `use_draft_interaction_features`
- `use_draft_conditional_behaviour_features`
- `draft_conditional_min_samples`
- `run_feature_group_ablation`
- `champion_scaling_method` (`avg_duration_delta`)
- `champion_scaling_smoothing`
- `champion_scaling_min_samples`
- `champion_scaling_recency_weighting`
- `champion_scaling_recency_half_life_days`
- `champion_scaling_patch_aware`
- `run_champion_scaling_ablation`
- `catboost_iterations`
- `catboost_early_stopping_rounds`
- `catboost_param_grid`
- `lgbm_param_grid` (used when `primary_model: lightgbm`)

Default setup is robust V1:
- target in seconds internally
- role-specific + bag-of-champions enabled
- rolling window size 10

## CLI Usage

### 1) Inspect schema

```bash
python main.py inspect --config config/example_config.yaml --output-report reports/schema_report.txt
```

### 2) Build match-level table only

```bash
python main.py flatten --config config/example_config.yaml --output-csv artifacts/match_level_table.csv
```

### 3) Train + evaluate + SHAP

```bash
python main.py train --config config/example_config.yaml
```

Outputs include:
- `artifacts/model_artifacts.joblib`
- `artifacts/match_level_table.csv`
- `reports/benchmark_summary.csv`
- `reports/metrics_summary.json`
- `reports/champion_scaling_ablation.csv` (with/without scaling for current primary model)
- `reports/feature_group_ablation.csv` + `reports/feature_group_ablation.json` (A→E feature-group ablation)
- `reports/champion_scaling_lookup.csv` (if scaling enabled)
- `artifacts/champion_scaling_lookup.joblib` (if scaling enabled)
- `reports/model_input_feature_columns.csv` + `reports/transformed_feature_columns.csv`
- `reports/catboost_feature_importance.csv` (when primary model is CatBoost)
- residual/error breakdown plots
- `reports/shap/` diagnostics

### 4) Inference on one draft payload

```bash
python main.py predict --artifact-path artifacts/model_artifacts.joblib --input-json sample_input.json --explain
```

Returned fields:
- `predicted_duration_seconds`
- `predicted_duration_minutes`
- optional SHAP explanation payload

### 5) Frontend app for upcoming-game predictions

Run the Streamlit app:

```bash
streamlit run frontend_app.py
```

In the app:
- choose your `model_artifacts.joblib`
- optionally prefill from a recent historical match template
- enter upcoming match metadata and full draft (or quick-paste roles as `top,jng,mid,bot,sup`)
- optionally auto-fill team priors from `artifacts/match_level_table.csv`
- click **Predict Match Duration**

## Deploy To Streamlit Community Cloud

1. Push this project to a GitHub repo.
2. Make sure these files are present in that repo:
   - `frontend_app.py`
   - `minutemodel/`
   - `requirements.txt`
   - `artifacts/model_artifacts.joblib`
   - optionally `artifacts/match_level_table.csv` for prior auto-fill
3. In Streamlit Community Cloud, click **Create app** and select:
   - Repository: your repo
   - Branch: your deployment branch (for example `main`)
   - Main file path: `frontend_app.py`
4. Deploy, then open the app URL.

Notes:
- If the build fails on `shap`, remove `shap` from `requirements.txt` (the app still runs; explanations become optional).
- Use forward-slash paths in the app configuration (`artifacts/model_artifacts.joblib`).

## Evaluation Metrics

Headline metric:
- MAE (minutes)

Secondary metrics:
- RMSE (minutes)
- Median Absolute Error (minutes)
- within-2-minute accuracy
- within-5-minute accuracy
- MAE (seconds) optional reference

## Benchmarking Guidance

Always interpret the primary model against baselines:
- global mean
- league mean
- league + patch mean
- ridge regression

If draft-only model gains are small, that still sets a useful pre-game benchmark.

## Important Modeling Context

Draft-only duration prediction is inherently noisy. It should not be treated as directly equivalent to models using in-game telemetry.

Expected progression:
- V1 (this repo): draft-only benchmark
- V2: add live-state features (for example 10-minute telemetry) for stronger predictability

## Extending to Version 2 (10-Minute Telemetry)

Suggested path:
1. Keep current match-level draft pipeline as static pre-game branch.
2. Add a telemetry branch with snapshot features at fixed game time (for example 10:00).
3. Train hybrid model: draft priors + telemetry state.
4. Evaluate with identical chronological protocol and baseline controls.

This preserves comparability while quantifying incremental value from live game state.
