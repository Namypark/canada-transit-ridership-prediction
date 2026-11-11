# Predicting Canadian Urban Transit Ridership in a Changing Fuel-Price Environment

CSCN8010 Extra-Credit Project

**Student:** Nnamdi

**Student ID:** 9085175

## Problem summary

**Question:** How can historical transit demand and gasoline-price information be combined to anticipate monthly Canadian urban transit ridership?

**Observable problem.** Monthly urban transit ridership in Canada changes with the seasons and has not returned to its pre-2020 level. This makes demand uncertain for anyone monitoring it.

- **Seasonal pattern:** In 2017-2019 and 2023-2024, July and August were below that year's average month every time. September and October were the highest on average. For example, trips rose from 146.2 million in August 2019 to 169.2 million in October 2019 (+15.7%).
- **Year-over-year change:** In April 2020, there were 25.7 million trips, compared with 157.3 million in April 2019 (-83.7%). Annual ridership in 2024 was 1,590.4 million trips, 84.2% of the 2019 total. In 2025, it was 1,552.6 million, or 82.2% of 2019.
- **Gasoline context:** Gasoline ranged from 77.8¢/L (April 2020) to 207.2¢/L (June 2022). Its association with ridership three months later was weak overall (r = 0.11) and changed by period, so it cannot be read as an effect.

**Approach.** Two Statistics Canada tables are validated, joined one-to-one by month, and turned into six leakage-safe predictors (lags at t-3 and t-12, gasoline at t-3, calendar position, and trend). A scikit-learn `Pipeline` (`StandardScaler` followed by `LinearRegression`) is trained on 2018-2024. It is then compared with a same-month-last-year baseline on the 12 months of 2025, with each forecast made three months ahead.

**Result.** The model did not improve on the baseline. Its MAE was 12.46 million trips (RMSE 15.11), compared with 5.40 million (RMSE 6.75) for the baseline. The model under-forecast 9 of the 12 months. The evaluation gives no evidence that gasoline prices improve these forecasts. The results are a historical experiment on a downloaded, potentially revised snapshot, not live forecasting performance.

## Data provenance

Both tables were downloaded on **November 11, 2026** using Statistics Canada's "CSV Download entire table" option. The original English CSVs and metadata files are preserved unmodified in `data/raw/`.

|                                  | Outcome                                                                                                                              | Potential contributing factor                                                                                                        |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| Table title                      | Passenger bus and urban transit statistics, by the North American Industry Classification System (NAICS)                             | Monthly average retail prices for gasoline and fuel oil, by geography                                                                |
| Table number                     | 23-10-0251-01                                                                                                                        | 18-10-0001-01                                                                                                                        |
| Table URL                        | [https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=2310025101](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=2310025101) | [https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000101](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000101) |
| Download URL                     | [https://www150.statcan.gc.ca/n1/tbl/csv/23100251-eng.zip](https://www150.statcan.gc.ca/n1/tbl/csv/23100251-eng.zip)                 | [https://www150.statcan.gc.ca/n1/tbl/csv/18100001-eng.zip](https://www150.statcan.gc.ca/n1/tbl/csv/18100001-eng.zip)                 |
| Selected series                  | GEO: Canada; NAICS: Urban transit systems [485110]; measure: Total passenger trips                                                   | GEO: Canada; fuel: Regular unleaded gasoline at self service filling stations                                                        |
| Published unit                   | Number, scaled in millions                                                                                                           | Cents per litre                                                                                                                      |
| Working variable                 | `trips_millions`                                                                                                                     | `gasoline_cents_per_litre`                                                                                                           |
| Survey                           | Monthly Passenger Bus and Urban Transit Survey (2745)                                                                                | Consumer Price Index (2301)                                                                                                          |
| Reference period in the download | January 2017 to August 2026                                                                                                          | January 1979 to September 2026                                                                                                       |
| Reference period used            | January 2017 to December 2025                                                                                                        | January 2017 to December 2025                                                                                                        |

Observations from 2026 are excluded by the project scope.

### Notes from the source metadata

- **Survey coverage:** Sampled companies account for at least 75% of provincial revenues for Canadian urban and inter-city transit (table 23-10-0251-01, note 1).
- **Corrections:** Data for January 2017 were corrected on May 5, 2017. On August 1, 2018, certain data points were suppressed because of data quality issues for May 2018. The selected passenger-trip series has no missing values in the project scope.
- **Status symbols:** Neither selected series has a `STATUS`, `SYMBOL`, or `TERMINATED` flag from January 2017 to December 2025.
- **Seasonal adjustment:** Neither series is seasonally adjusted, which is why the seasonal pattern is visible in the data.
  - **Passenger trips:** The Monthly Passenger Bus and Urban Transit Survey states, "The data are not seasonally adjusted" ([survey 2745](https://www23.statcan.gc.ca/imdb/p2SV.pl?Function=getSurvey&SDDS=2745)).
  - **Gasoline prices:** Table 18-10-0001-01 does not state this directly. However, the Consumer Price Index program seasonally adjusts only 15 national series, and average retail gasoline prices are not among them ([survey 2301](https://www23.statcan.gc.ca/imdb/p2SV.pl?Function=getSurvey&SDDS=2301)).
- **Revisions:** Statistics Canada can revise published data, so this analysis uses the preserved snapshot below.

### Snapshot checksums (SHA-256)

| File                    | SHA-256                                                            |
| ----------------------- | ------------------------------------------------------------------ |
| `23100251.csv`          | `abb18c8855a71bc0ca47332b1d0fcfe4afd9df440b9f0dda1ac1ba36df1b4da0` |
| `23100251_MetaData.csv` | `2e5354950712cd43166357c3117e6df4d8ad61a7cc2ed6719827e800b78f4961` |
| `18100001.csv`          | `3f0ea8a5ec61d1272e118fff4b6a51b07aed7e55b9464db330bb36ec59d47818` |
| `18100001_MetaData.csv` | `7ce72d24d08b725a4519ec4be45a19b74beaea470aca8de94b65bdd16c0fa4ce` |

## Project contents

```
.
├── Canada_Transportation_Prediction.ipynb   All stages, explanations, outputs, and the graphical report
├── README.md
├── requirements.txt                         Exact package versions
├── data/
│   ├── raw/                                 Original Statistics Canada CSVs and metadata
│   └── processed/
│       ├── monthly_transportation.csv       Joined monthly dataset
│       └── model_data.csv                   Modelling dataset
└── outputs/
    └── predictions_2025.csv                 Test-period predictions and errors
```

## Environment

Python 3.14.2

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Figures are built with Plotly. They are interactive in Jupyter and VS Code, and each one also embeds a static PNG copy so it still appears in viewers without interactive output, such as GitHub's notebook preview. The PNG copies are made by `kaleido`, which needs a Chrome browser. If Chrome is not installed, run `plotly_get_chrome` once inside the environment. Without it, the notebook still runs, and the figures are interactive only.

## Run instructions

From the repository root, with the environment active, run all cells in order:

```bash
jupyter nbconvert --to notebook --execute --inplace Canada_Transportation_Prediction.ipynb
```

Alternatively, open the notebook with `jupyter lab`, then choose **Kernel → Restart Kernel and Run All Cells**.

The notebook recreates the processed data, predictions, and figures from `data/raw/`. It needs no credentials or manual code edits.
