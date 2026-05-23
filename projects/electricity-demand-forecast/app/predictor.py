from pathlib import Path
import os
import pickle
import sys

import pandas as pd
import numpy as np
import lightgbm as lgb
import yaml


def load_config(ROOT:str)-> dict:

    path_config = os.path.join(ROOT, "config/config.yaml")

    with open(path_config, "rb") as f:
        config = yaml.safe_load(f)
    
    return config


def load_model(ROOT:str) -> lgb.LGBMRegressor:

    # 現状、一個のモデルを読む。
    path_model = os.path.join(ROOT, "model/model.pkl")

    with open(path_model, "rb") as f:
        model = pickle.load(f)
    
    return model


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
     
     # weather(天候：晴れとか、曇りとか）をカテゴリカルデータに変更
    for col_name in df.filter(like="weather").columns:
            df[col_name] = df[col_name].astype("category")

    return df


def predict(data:list) -> list:

    # 予測に必要なデータを読み込む際に、ローカルとdockerで共通の引数を扱う。パス問題解決のためにルートを取得
    ROOT = Path(__file__).resolve().parent
    config = load_config(ROOT=ROOT)
    model = load_model(ROOT=ROOT)

    df = pd.DataFrame(data)

    df = df.replace({None: np.nan})

    # 学習時のカラムの順番に並び替える
    df = df[config["input_data"]["feature_order"]]
     
    #  カテゴリー型に変更
    df = preprocessing(df=df)

    print("pred starts")
    y_pred = model.predict(df)
    print("pred_y:", y_pred[:10])

    return y_pred.tolist()
    



# predictor.py単体での動作確認用

def main():

    print("current_file_path", Path(__file__).resolve())
    file_path = os.path.join(Path(__file__).resolve().parent, "data/X_test.csv")
    df = pd.read_csv(file_path)
    
    dict_test = {
        "request id": "test",
        "data": df.to_dict("records")
        }
    

    predict(dict_test["data"])


if __name__ == "__main__":

    main()