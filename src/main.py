from typing import Any

from pydantic.types import Json
from feature_extractor import FeatureExtractor
import uvicorn
from fastapi import FastAPI, HTTPException
import nest_asyncio
from pydantic import BaseModel
import json

from res_controller import inspection, base64toImage

app = FastAPI(debug=True)
nest_asyncio.apply()

# Load the feature extractor before anything
fe = FeatureExtractor()


class qImage(BaseModel):
    img: str


# Proof of concept work that receives a base64 image
# and returns the extracted feature vector.
@app.post("/extract")
def extractFeature(qImage: qImage):
    q = qImage.img
    res = fe.extract(base64toImage(q))
    return res.tolist()


uvicorn.run(app, host="0.0.0.0", port=7000)
