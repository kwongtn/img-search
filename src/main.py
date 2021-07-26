import env_loader

from typing import Any
from feature_extractor import FeatureExtractor
import uvicorn
from fastapi import FastAPI, HTTPException
import nest_asyncio
from pydantic import BaseModel
import datetime
import numpy as np
import os

from utils import base64toImage

app = FastAPI(debug=True)
nest_asyncio.apply()

# Load the feature extractor before anything
fe = FeatureExtractor()

# ===================================================


class qImage(BaseModel):
    img: str


class AddImageClass(BaseModel):
    img: str
    id: str
    attribs: Any


# Receives a base64 image
# and returns the extracted feature vector.
@app.post("/extract")
def extractFeature(qImage: qImage):
    q = qImage.img
    res = fe.extract(base64toImage(q))
    return res.tolist()


class qTwoImg(BaseModel):
    img1: str
    img2: str

# Compare between both images via extracted features


@app.post("/compare")
def compare(qTwoImg: qTwoImg):
    q1 = fe.extract(base64toImage(qTwoImg.img1))
    q2 = fe.extract(base64toImage(qTwoImg.img2))

    # Get eucledian distance between the two vectors
    dists = np.linalg.norm(q1 - q2)

    res = {}
    res["distance"] = dists.item()

    return res


@app.get("/healthcheck")
async def healthcheck():
    return datetime.datetime.now()

if __name__ == "__main__":
    uvicorn.run("main:app",
                host=str(os.environ.get("SERVER_ADDRESS")),
                port=int(os.environ.get("SERVER_PORT")),
                log_level=str(os.environ.get("SERVER_LOG_LEVEL")),
                reload=bool(os.environ.get("SERVER_LIVE_RELOAD"))
                )


# @app.on_event("startup")
# async def on_startup():
#     app.state.executor = ProcessPoolExecutor()


@app.on_event("shutdown")
async def on_shutdown():
    app.state.executor.shutdown()
