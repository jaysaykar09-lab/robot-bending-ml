from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
import pandas as pd
from io import StringIO
from sklearn.ensemble import RandomForestRegressor

app = FastAPI(title="Robot Bending ML API", version="1.0")

# Allow browser frontend to call this API (local demo).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for local testing; restrict for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Embedded dataset (your CSV)
# Forward model: ryr_rad -> final_angle_deg
# -----------------------------
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
11,0,0,0,0,1.05,0,66.95,9.06
12,0,0,0,0,1.1,0,69.81,9.04
13,0,0,0,0,1.15,0,75.26,9.03
14,0,0,0,0,1.25,0,80.86,8.95
15,0,0,0,0,1.3,0,81.98,8.4
16,0,0,0,0,1.35,0,81.8,7.93
17,0,0,0,0,1.45,0,82.12,7.55
18,0,0,0,0,1.5,0,82.08,7.49
19,0,0,0,0,1.55,0,81.45,7.44
20,0,0,0,0,1.6,0,81.68,7.43
21,0,0,0,0,1.65,0,81.69,7.26
22,0,0,0,0,1.7,0,81.43,7.19
23,0,0,0,0,1.75,0,81.51,7.06
24,0,0,0,0,1.8,0,81.68,6.96
25,0,0,0,0,1.85,0,81.75,6.87
26,0,0,0,0,1.9,0,81.97,6.8
27,0,0,0,0,1.95,0,82.09,6.74
28,0,0,0,0,2.0,0,82.1,6.7
29,0,0,0,0,2.05,0,82.11,6.67
30,0,0,0,0,2.1,0,82.12,6.64
31,0,0,0,0,2.15,0,82.12,6.63
32,0,0,0,0,2.2,0,82.13,6.61
"""

df = pd.read_csv(StringIO(CSV_DATA)).rename(
    columns={"ryr (rad)": "ryr_rad", "Final Angle ? (deg)": "final_angle_deg"}
)

X = df[["ryr_rad"]]
y = df["final_angle_deg"]

model = RandomForestRegressor(n_estimators=400, max_depth=12, random_state=42)
model.fit(X, y)

RYR_MIN = float(df["ryr_rad"].min())
RYR_MAX = float(df["ryr_rad"].max())

def invert_desired_angle(desired_angle_deg: float, steps: int = 4000):
    """
    Find ryr_rad that makes predicted angle closest to desired.
    Uses forward model + grid search (stable inverse).
    """
    grid = np.linspace(RYR_MIN, RYR_MAX, steps).reshape(-1, 1)
    pred_angles = model.predict(grid)
    idx = int(np.argmin(np.abs(pred_angles - desired_angle_deg)))
    best_ryr = float(grid[idx][0])
    achieved = float(pred_angles[idx])
    err = abs(achieved - desired_angle_deg)
    return best_ryr, achieved, err

class PredictRequest(BaseModel):
    desired_angle: float

@app.get("/")
def root():
    return {"status": "ok", "message": "Robot Bending ML API running"}

@app.post("/predict")
def predict(req: PredictRequest):
    desired = float(req.desired_angle)
    best_ryr, achieved, err = invert_desired_angle(desired)

    # In your dataset rx/ry/rz/rxr/rzr are 0 always, so returning 0 is correct.
    return {
        "desired_angle_deg": desired,
        "rx_mm": 0.0,
        "ry_mm": 0.0,
        "rz_mm": 0.0,
        "rxr_rad": 0.0,
        "ryr_rad": best_ryr,
        "rzr_rad": 0.0,
        "achieved_angle_deg": achieved,
        "abs_error_deg": err,
        "ryr_range_rad": [RYR_MIN, RYR_MAX],
    }
