import os, joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_PATH = "CRS_dataset_final_with_names.csv"


OUT_DIR = os.environ.get('OUT_DIR', './model')
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Drop obviously non-predictive columns if present
drop_cols = [c for c in ['name','id','first_name','last_name','email'] if c in df.columns]
df = df.drop(columns=drop_cols, errors='ignore')

# Target
target_col = 'career_aspiration' if 'career_aspiration' in df.columns else None
if target_col is None:
    raise RuntimeError('Target column career_aspiration not found in dataset')

y = df[target_col].astype(str)
X = df.drop(columns=[target_col])

# Coerce booleans
for b in ['extracurricular_activities','part_time_job']:
    if b in X.columns:
        X[b] = X[b].astype(str).str.lower().isin(['true','1','yes','y'])

# Identify numeric and categorical columns
numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
categorical_cols = [c for c in X.columns if c not in numeric_cols]

pre = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced_subsample')
pipe = Pipeline(steps=[('pre', pre), ('clf', clf)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pipe.fit(X_train, y_train)

print(classification_report(y_test, pipe.predict(X_test)))

# Save model and classes
joblib.dump(pipe, os.path.join(OUT_DIR, 'model.joblib'))
np.save(os.path.join(OUT_DIR, 'classes.npy'), pipe.classes_)
print('Saved to', OUT_DIR)
