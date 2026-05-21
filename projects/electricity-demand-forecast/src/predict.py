from pathlib import Path
import os
import pickle
import sys

import pandas as pd
import numpy as np
import lightgbm as lgb


def load_data(path: str) -> pd.DataFrame:
    
    df = pd.read_csv(path)

    return df


def load_model(path: str) -> lgb.LGBMRegressor:

    with open(path, "rb") as f:
        model = pickle.load(f)
    
    return model


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
     
     # weather(天候：晴れとか、曇りとか）をカテゴリカルデータに変更
    for col_name in df.filter(like="weather").columns:
            df[col_name] = df[col_name].astype("category")

    return df


def predict(model: lgb.LGBMRegressor, df: pd.DataFrame) -> None: 

    df = preprocessing(df=df)

    print("pred starts")
    y_pred = model.predict(df)

    print("pred_y:", y_pred[:10])
    

def main():

    # docker動作時にパスの確認用。
    # print("カレントディレクトリ", os.getcwd())

    # 引数チェック
    if len(sys.argv) < 2:
        raise ValueError("input path required")

    # 予測に必要なデータを読み込む際に、ローカルとdockerで共通の引数を扱う。パス問題解決のためにルートを取得
    ROOT = Path(__file__).resolve().parent.parent
    # print("root: ", ROOT)
    path_input = os.path.join(ROOT, sys.argv[1])

    # 現状、一個のモデルを読む。
    path_model = os.path.join(ROOT, "models", "model.pkl")

    # 引数でパス指定。
    df = load_data(path=path_input)
    # df = load_data(path="../data/processed/dataset/X_test.csv")
  
    # 引数でパス指定。
    model = load_model(path=path_model)
    # model = load_model(path="../models/model.pkl")

    predict(model=model, df=df)


if __name__ == "__main__":
    main()