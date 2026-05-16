# ml-lab

機械学習・MLOps・LLM 周辺技術の軽量実験リポジトリ。

portfolio リポジトリとは別に、  
「技術を試す場所」として運用しています。

---

# Purpose

- MLOps 技術に触れる
- Docker / AWS / MLflow などの周辺技術を試す
- 学習ログ・試行錯誤を残す

完成度の高い portfolio というより、

- 実験
- 検証
- 学習記録
- 試作

を目的とした repository です。

---

# Projects

## 1. electricity-demand-forecast

東京電力の公開データを用いた電力需要予測。

### What I tried

- LightGBM による時系列予測
- lag / rolling 特徴量
- sin / cos による周期特徴量
- SHAP によるモデル解釈
- MLflow による experiment tracking

### Current status

- baseline → feature engineering により精度改善
- MLflow による実験管理を導入
- Docker 化 / SageMaker deploy を今後実施予定

📁 `electricity-demand-forecast/`

---

# Tech Stack

- Python
- pandas
- scikit-learn
- LightGBM
- MLflow
- SHAP

---

# Notes

- Notebook ベースで軽量に実験
- 試行錯誤の過程も一部含みます