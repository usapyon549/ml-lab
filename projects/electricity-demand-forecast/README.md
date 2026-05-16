# electricity-demand-forecast

東電管轄の電力需要実績値を用いた電力需要予測。

## What I tried

- LightGBM
- feature engineering
- MLflow
- SHAP analysis

---

## Data source

下記の公開データを使用。

### 気象データ
気象庁  
https://www.data.jma.go.jp/risk/obsdl/index.php

### 電力需要データ
東京電力パワーグリッド  
https://www.tepco.co.jp/forecast/html/area_data-j.html

---

## Notes

- 気温依存性が強い
- 曜日特徴量が有効
- lag / rolling 特徴量が有効
- 気温・時間特徴量は、feature importance / SHAP の両方で寄与が大きい
- 気象特徴量には、有効なものと寄与の小さいものが存在

---

## Result

MAE：

```text
baseline: 2176
improved: 1093