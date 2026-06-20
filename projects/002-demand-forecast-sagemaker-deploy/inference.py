import os
import pickle
import json

import pandas as pd
import numpy as np
import lightgbm as lgb


# 学習時のカラムの順番に並び替える(エンドポイントの動作確認優先のため直書)
# TODO: config.yamlに分けて管理する。
feature_order = [
    "chiba_temperature",
    "gumma_temperature",
    "ibaraki_temperature",
    "kanagawa_temperature",
    "saitama_temperature",
    "tochigi_temperature",
    "tokyo_temperature",
    "yamanashi_temperature",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "chiba_temperature_lag1",
    "chiba_temperature_lag2",
    "chiba_temperature_lag3",
    "chiba_temperature_delta_lag1",
    "gumma_temperature_lag1",
    "gumma_temperature_lag2",
    "gumma_temperature_lag3",
    "gumma_temperature_delta_lag1",
    "ibaraki_temperature_lag1",
    "ibaraki_temperature_lag2",
    "ibaraki_temperature_lag3",
    "ibaraki_temperature_delta_lag1",
    "kanagawa_temperature_lag1",
    "kanagawa_temperature_lag2",
    "kanagawa_temperature_lag3",
    "kanagawa_temperature_delta_lag1",
    "saitama_temperature_lag1",
    "saitama_temperature_lag2",
    "saitama_temperature_lag3",
    "saitama_temperature_delta_lag1",
    "tochigi_temperature_lag1",
    "tochigi_temperature_lag2",
    "tochigi_temperature_lag3",
    "tochigi_temperature_delta_lag1",
    "tokyo_temperature_lag1",
    "tokyo_temperature_lag2",
    "tokyo_temperature_lag3",
    "tokyo_temperature_delta_lag1",
    "yamanashi_temperature_lag1",
    "yamanashi_temperature_lag2",
    "yamanashi_temperature_lag3",
    "yamanashi_temperature_delta_lag1",
    "chiba_temperature_sma3",
    "chiba_temperature_sma5",
    "chiba_temperature_delta_sma3",
    "chiba_temperature_delta_sma5",
    "gumma_temperature_sma3",
    "gumma_temperature_sma5",
    "gumma_temperature_delta_sma3",
    "gumma_temperature_delta_sma5",
    "ibaraki_temperature_sma3",
    "ibaraki_temperature_sma5",
    "ibaraki_temperature_delta_sma3",
    "ibaraki_temperature_delta_sma5",
    "kanagawa_temperature_sma3",
    "kanagawa_temperature_sma5",
    "kanagawa_temperature_delta_sma3",
    "kanagawa_temperature_delta_sma5",
    "saitama_temperature_sma3",
    "saitama_temperature_sma5",
    "saitama_temperature_delta_sma3",
    "saitama_temperature_delta_sma5",
    "tochigi_temperature_sma3",
    "tochigi_temperature_sma5",
    "tochigi_temperature_delta_sma3",
    "tochigi_temperature_delta_sma5",
    "tokyo_temperature_sma3",
    "tokyo_temperature_sma5",
    "tokyo_temperature_delta_sma3",
    "tokyo_temperature_delta_sma5",
    "yamanashi_temperature_sma3",
    "yamanashi_temperature_sma5",
    "yamanashi_temperature_delta_sma3",
    "yamanashi_temperature_delta_sma5",
    "chiba_precipitation",
    "gumma_precipitation",
    "ibaraki_precipitation",
    "kanagawa_precipitation",
    "saitama_precipitation",
    "tochigi_precipitation",
    "tokyo_precipitation",
    "yamanashi_precipitation",
    "chiba_sunshine_duration",
    "gumma_sunshine_duration",
    "ibaraki_sunshine_duration",
    "kanagawa_sunshine_duration",
    "saitama_sunshine_duration",
    "tochigi_sunshine_duration",
    "tokyo_sunshine_duration",
    "yamanashi_sunshine_duration",
    "chiba_dew_point",
    "gumma_dew_point",
    "ibaraki_dew_point",
    "kanagawa_dew_point",
    "saitama_dew_point",
    "tochigi_dew_point",
    "tokyo_dew_point",
    "yamanashi_dew_point",
    "chiba_vapor_pressure",
    "gumma_vapor_pressure",
    "ibaraki_vapor_pressure",
    "kanagawa_vapor_pressure",
    "saitama_vapor_pressure",
    "tochigi_vapor_pressure",
    "tokyo_vapor_pressure",
    "yamanashi_vapor_pressure",
    "chiba_humidity",
    "gumma_humidity",
    "ibaraki_humidity",
    "kanagawa_humidity",
    "saitama_humidity",
    "tochigi_humidity",
    "tokyo_humidity",
    "yamanashi_humidity",
    "chiba_weather",
    "gumma_weather",
    "ibaraki_weather",
    "kanagawa_weather",
    "saitama_weather",
    "tochigi_weather",
    "tokyo_weather",
    "yamanashi_weather",
    "chiba_visibility",
    "gumma_visibility",
    "ibaraki_visibility",
    "kanagawa_visibility",
    "saitama_visibility",
    "tochigi_visibility",
    "tokyo_visibility",
    "yamanashi_visibility"
]


#-------------
# utils
#-------------

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
     
     # weather(天候：晴れとか、曇りとか）をカテゴリカルデータに変更
    for col_name in df.filter(like="weather").columns:
            df[col_name] = df[col_name].astype("category")

    # json API 経由の、nan/null対策で、カテゴリーカラム以外はnumericにする。
    for col_name in df.columns:
        if "weather" not in col_name:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")


    return df

# -------------
# for sagemaker
# -------------

def model_fn(model_dir):

    # S3に保存されたモデルを読み込む。
    model_path = os.path.join(model_dir, "model.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    return model


def input_fn(request_body, request_content_type):

    if request_content_type == "application/json":
        request_body = json.loads(request_body)
        return pd.DataFrame(request_body["inputs"])
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):

    df = input_data

    # api経由で受け取ったjsonのnan/null処理
    df = df.replace({None: np.nan})

    df = df[feature_order]
     
    #  カテゴリー型に変更
    df = preprocessing(df=df)

    return model.predict(df)


def output_fn(prediction, content_type):
    return json.dumps({"predictions": prediction.tolist()}), content_type



# ------------------------
#  ローカルでの動作確認用
# ------------------------

if __name__ == "__main__":

    print("ローカルテスト開始")

    model = model_fn(model_dir = "./model")
    print("モデル読み込み完了")

    df = pd.read_csv("./data/X_test.csv")

    # ダミーボディーの作成。目視確認しやすいように、head(2)
    dummy_body = {
        "inputs": df.head(2).replace({np.nan: None}).to_dict("records")
    }

    inputs = input_fn(json.dumps(dummy_body), "application/json")

    print("予測開始")
    preds = predict_fn(inputs, model)

    print("preds: ", preds)
