from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from SentimentAnalyzer import *
import pandas as pd
import tempfile
import os

app = FastAPI()
analyzer = SentimentAnalyzer()

class SentimentInput(BaseModel):
    text: str

class SentimentResult(BaseModel):
    sentiment: str
    confidence: float
    trigger: str

@app.get("/")
async def get_html():
    return FileResponse("index.html")

@app.post("/predict_sentiment_text/")
async def predict_sentiment_text(sentiment_input: SentimentInput):
    text = sentiment_input.text
    sentiment_pred, sentiment_conf, trigger = analyzer.sentiment_classify(text)
    return {"sentiment": sentiment_pred, "confidence": sentiment_conf, "trigger": trigger}

@app.post("/predict_sentiment_excel/")
async def predict_sentiment_excel(file: UploadFile = File(...)):
    if not file.filename.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Only Excel files (.xlsx) are supported.")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    df = pd.read_excel(tmp_path)

    if "text" not in df.columns:
        os.remove(tmp_path)
        raise HTTPException(status_code=400, detail="Input Excel file must contain a column named 'text'.")

    results = []
    for idx, row in df.iterrows():
        text = row["text"]
        sentiment_pred, sentiment_conf, trigger = analyzer.sentiment_classify(text)
        results.append({"sentiment": sentiment_pred, "confidence": sentiment_conf, "trigger": trigger})

    df_result = pd.concat([df, pd.DataFrame(results)], axis=1)

    output_filename = "sentiment_analysis_result.xlsx"
    df_result.to_excel(output_filename, index=False)

    os.remove(tmp_path)

    return FileResponse(output_filename, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=output_filename)
