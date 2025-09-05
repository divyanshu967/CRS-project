# Career Recommendation System (Flask + ML)

## Features
- User auth (login/signup) with hashed passwords
- Profile form (scores, interests, skills, preferences)
- ML model (RandomForest) trained on your CSV
- Results with top career matches + Chart.js visualization
- Bootstrap dark UI
- Ready for local run and Vercel deploy

## Quickstart (Local)
1) Create a virtual env and install deps:
```
pip install -r requirements.txt
```
2) Train the model (uses the uploaded CSV by default path):
```
python train_model.py
```
This writes `model/model.joblib` and `model/classes.npy`.

3) Run the app:
```
python api/index.py
```
Open http://127.0.0.1:5000

## Using your own dataset path
Edit `DATA_PATH` env var:
```
DATA_PATH="path/to/CRS_dataset_final_with_names.csv" python train_model.py
```

## Deploy to Vercel
- Install Vercel CLI: `npm i -g vercel`
- From project root:
```
vercel dev
```
- Then:
```
vercel
```
Vercel uses `vercel.json` to route all requests to `api/index.py`.

> Tip: Set `SECRET_KEY` in Vercel Project Settings for sessions.

## Files
- api/index.py — Flask app (routes, login, predict)
- train_model.py — trains model and saves artifacts
- model/ — saved model files
- templates/ — HTML
- static/css/style.css — styling
- requirements.txt — dependencies
- vercel.json — Vercel config

## Notes
- The training pipeline auto-detects numeric/categorical columns and one-hot encodes cats.
- For fairness, consider dropping `gender` or applying debiasing.
- For better near/long-term advice, augment text templates or add a rules layer.
