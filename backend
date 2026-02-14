from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.ensemble import RandomForestRegressor

app = FastAPI(title="Robot Bending ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CSV_DATA = """Run ID,rx (mm),ry (mm),rz (mm),rxr (rad),ryr (rad),rzr (rad),Final Angle ? (deg),Radius
0,0,0,0,0,1.0,0,61.2,9.1
1,0,0,0,0,1.2,0,80.34,8.95
2,0,0,0,0,1.4,0,82.1,7.7
3,0,0,0,0,0.6,0,31.63,15.86
4,0,0,0,0,0.65,0,35.14,14.55
5,0,0,0,0,0.7,0,40.47,13.32
6,0,0,0,0,0.75,0,44.86,12.24
7,0,0,0,0,0.8,0,48.35,11.31
8,0,0,0,0,0.85,0,50.79,10.58
9,0,0,0,0,0.9,0,53.11,10.07
10,0,0,0,0,0.95,0,56.6,9.58
"""

df = pd.read_csv(StringIO(CSV_DATA))
df = df.rename(columns={"ryr (rad)": "ryr", "Final Angle ? (deg)": "angle"})

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(df[["ryr"]], df["angle"])

RYR_MIN, RYR_MAX = df["ryr"].min(), df["ryr"].max()

def invert(desired):
    grid = np.linspace(RYR_MIN, RYR_MAX, 3000).reshape(-1, 1)
    preds = model.predict(grid)
    i = np.argmin(np.abs(preds - desired))
    return float(grid[i][0]), float(preds[i])

class Req(BaseModel):
    desired_angle: float

@app.post("/predict")
def predict(req: Req):
    ryr, achieved = invert(req.desired_angle)
    return {
        "ryr_rad": ryr,
        "achieved_angle_deg": achieved,
        "abs_error_deg": abs(achieved - req.desired_angle)
    }
