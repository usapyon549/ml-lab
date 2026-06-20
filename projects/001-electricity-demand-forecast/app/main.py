from fastapi import FastAPI
import pandas as pd

from app.predictor import predict


app = FastAPI()

# GETでpingがきたら、FastAPIにping()が渡される
# デコレータ
@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/invocations")
def invocations(request: dict):

    if "data" not in request:
        return {"error": "data key missing"}
    
    result = predict(request["data"])

    return {
        "prediction":result
    }
