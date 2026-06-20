# electricity-demand-forecast

東電管轄の電力需要実績値を用いた電力需要予測。

## What I tried

- LightGBM
- feature engineering
- MLflow
- SHAP analysis
- fastAPI
- Docker

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

## 予測モデル精度

MAE：

```text
baseline: 2176
improved: 1093
```

---

## Notes

### 電力需要予測モデル
- 気温依存性が強い
- sin/cos化が有効
- 曜日特徴量が有効
- lag / rolling 特徴量が有効
- 気温・時間特徴量は、feature importance / SHAP の両方で寄与が大きい
- 気象特徴量には、有効なものと寄与の小さいものが存在

### 予測モデルのAPI/コンテナ化
- FastAPI により推論 API を実装
- Docker により推論環境をコンテナ化
- /ping、/invocations endpoint を実装
- Notebook → HTTP request → model prediction の動作を確認
- pathlib.Path により、ローカル / Docker 間の path 差異を吸収
- 学習時の feature 順序を config 管理し、推論時に統一
- JSON 経由での NaN / null handling を実装
- 軽量 image (python:3.12-slim) を利用
- LightGBM 実行に必要な OS library (libgomp1) を追加


