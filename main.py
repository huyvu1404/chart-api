from fastapi import FastAPI
from api.endpoints import chart

app = FastAPI(title="Chart API")

app.include_router(chart, prefix="/chart", tags=["Chart"])
