import os
import pickle

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
    print("カレントディレクトリ", os.getcwd())

    # df = load_data(path="../data/processed/dataset/X_test.csv")
    # docker 用。  
    df = load_data(path="data/processed/dataset/X_test.csv")
  
    # model = load_model(path="../models/model.pkl")
    # docker用。
    model = load_model(path="models/model.pkl")

    predict(model=model, df=df)


if __name__ == "__main__":
    main()