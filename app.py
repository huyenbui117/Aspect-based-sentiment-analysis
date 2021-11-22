import fastapi
from fastapi import Depends, FastAPI, Response, status, File, UploadFile
from fastapi.responses import FileResponse
from main import run
import pandas as pd
app = fastapi.FastAPI()


@app.post(path="/demo", response_class=FileResponse)
async def main(file: UploadFile = File(media_type='multipart', default='Any')):
    file_location = "data/text.xlsx"
    with open(file_location, "wb") as file_object:
        file_object.write(file.file.read())
    df = pd.read_excel(file_location)
    df.to_csv("data/text.csv")
    run()
    return "data/predict_text.json"