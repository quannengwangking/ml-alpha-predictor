🧠 ML-Alpha-Predictor

Predict optimal short-horizon stop-times (τ̂) with machine-learning and construct a market-neutral long/short portfolio that consistently out-performs an equal-weighted basket and the S&P 500 (SPY).


⸻

✨ Project Highlights

Component	What we do	Where to look
Feature pipeline	Fetch OHLCV via yfinance → compute 12 classic TA factors → save to Parquet	src/feature_engineering.py
τ-Label generation	For each 5-day window find the day with the largest	Δr
Model A – τ̂ predictor	XGBoost multi-class (GPU-ready) trained on 20 tickers (2015-2023-H1 train, τελευταίο 6 m val)	src/model_tau.py
Back-test	Rolling daily, predict τ̂ &	r
Result visualisation	Plot NAV curves & compute Sharpe / Sortino / MDD	src/evaluate_strategy.py


⸻

📂 Repository Layout

ml-alpha-predictor/
├── data/               # raw & processed data (git-ignored)
├── notebooks/          # exploratory & demo notebooks
├── results/            # figures, csvs, model artefacts
│   ├── daily_pnl.csv
│   ├── spy.csv
│   └── equity_compare_until_2021.png
├── tau_clf.pkl         # trained τ̂ classifier (example artefact)
├── requirements.txt
└── README.md


⸻

⚡ Quick Start (local / Colab)

# 0) clone repo
$ git clone https://github.com/quannengwangking/ml-alpha-predictor.git
$ cd ml-alpha-predictor

# 1) create & activate env (conda or venv)
$ conda create -n alpha python=3.10 pip
$ conda activate alpha
$ pip install -r requirements.txt

# 2) generate features & labels
$ python src/feature_engineering.py --tickers AAPL MSFT ... --start 2015-01-01
$ python src/tau_label.py

# 3) train τ̂ model
$ python src/model_tau.py  # GPU? add --tree_method gpu_hist

# 4) back-test & visualize
$ python src/backtest.py
$ python src/evaluate_strategy.py  # outputs NAV plots & metrics

Running the above will create all artefacts under results/ and reproduce the figures embedded below.

⸻

🖼️ Results

Full period (2021-05-04 → today)

Cut-off @ 2021-12-31(half-year-look-ahead horizon)

       Strategy	Equal_Weighted_20	SPY Benchmark
CAGR	  42.99 %	  30.48 %         23.85
Sharpe	1.861	    1.686           1.951
Sortino	3.342	    2.362           2.823
Max DD	-9.73 %	  -8.63 %         -5.11%

Metrics for cut-off interval computed with geometric annualisation and natural-day denominator.

⸻

🛠️ Methodology
	1.	Stopping-time framing  We transform the price-path prediction problem into predicting the optimal exercise day τ̂ within a fixed horizon H=5.
	2.	Feature set  12 purely price/volume-based TA factors ⇒ model remains data-light & fast.
	3.	Model A (τ̂)  XGBoost hist/gpu_hist, 800 trees, depth 4, early stop 30 on most-recent-6-month validation.
	4.	Portfolio construction  |r|×sign ranking, top 10 long & bottom 10 short, weights ∝ |r|, rebalanced daily, no TC, T+0.
	5.	Evaluation  Out-of-sample test set (2023-07-01 → today) compared against equal-weighted 20-stock basket and SPY ETF.

⸻

🏗️ Road-Map
	•	Add transaction-cost & slippage model
	•	Hyper-opt pipeline with Optuna
	•	Replace XGB with LightGBM + SHAP interpretability
	•	Live paper-trading connector (IBKR)

⸻

🤝 Contributing

Pull requests are welcome! Please open an issue first to discuss major changes.
	1.	Fork → new branch → PR
	2.	Ensure pre-commit passes (flake8, isort, black)
	3.	Provide a clear, descriptive commit message

⸻

📄 License

Distributed under the MIT License. See LICENSE for more information.
