import os, sqlite3, json, numpy as np, pandas as pd, joblib
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__, static_folder='../static', template_folder='../templates')
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret')

# SQLite (file-based)
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'users.db')

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, name, email, password_hash):
        self.id = id
        self.name = name
        self.email = email
        self.password_hash = password_hash

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL
    )''')
    con.commit()
    con.close()

def get_user_by_email(email):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute('SELECT id, name, email, password_hash FROM users WHERE email=?', (email,))
    row = cur.fetchone()
    con.close()
    return User(*row) if row else None

def get_user_by_id(user_id):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute('SELECT id, name, email, password_hash FROM users WHERE id=?', (user_id,))
    row = cur.fetchone()
    con.close()
    return User(*row) if row else None

@login_manager.user_loader
def load_user(user_id):
    return get_user_by_id(user_id)

# Load ML pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.joblib')
CLASSES_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'classes.npy')
pipeline = None
classes_ = None
if os.path.exists(MODEL_PATH):
    pipeline = joblib.load(MODEL_PATH)
if os.path.exists(CLASSES_PATH):
    classes_ = np.load(CLASSES_PATH, allow_pickle=True)

@app.route('/')
def home():
    return redirect(url_for('input_form')) if current_user.is_authenticated else redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    init_db()
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']
        user = get_user_by_email(email)
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('input_form'))
        flash('Invalid email or password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    init_db()
    if request.method == 'POST':
        name = request.form['name'].strip()
        email = request.form['email'].strip().lower()
        password_hash = generate_password_hash(request.form['password'])
        try:
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            cur.execute('INSERT INTO users (name, email, password_hash) VALUES (?,?,?)',
                        (name, email, password_hash))
            con.commit()
            con.close()
            flash('Account created. Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already registered.', 'warning')
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/form')
@login_required
def input_form():
    return render_template('form.html')

def _coerce_bool(v):
    s = str(v).strip().lower()
    return True if s in ('true','1','yes','y') else False
def _prepare_features(form):
    # Convert form data to DataFrame
    df = pd.DataFrame({k: [v] for k, v in form.items()})

    # Fill missing columns expected by the pipeline
    if hasattr(pipeline, 'named_steps') and 'pre' in pipeline.named_steps:
        for name, trans, cols in pipeline.named_steps['pre'].transformers_:
            for col in cols:
                if col not in df.columns:
                    df[col] = 0.0 if name == 'num' else 'missing'

    # Numeric columns -> numeric, fill NaNs with 0
    numeric_cols = []
    for name, trans, cols in pipeline.named_steps['pre'].transformers_:
        if name == 'num':
            numeric_cols.extend(cols)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Categorical columns -> string, replace None/NaN/empty with 'missing'
    categorical_cols = []
    for name, trans, cols in pipeline.named_steps['pre'].transformers_:
        if name != 'num':
            categorical_cols.extend(cols)
    for col in categorical_cols:
        df[col] = df[col].apply(lambda x: str(x).strip() if x not in [None, np.nan, ''] else 'missing')

    # Reorder columns as expected by the pipeline
    return df[pipeline.feature_names_in_]



@app.route('/predict', methods=['POST'])
@login_required
def predict():
    global pipeline, classes_
    if pipeline is None or classes_ is None:
        flash('Model not loaded. Train the model first.', 'danger')
        return redirect(url_for('input_form'))
    df = _prepare_features(request.form)
    proba = pipeline.predict_proba(df)[0]
    # Build response: top 5 careers
    idx = np.argsort(proba)[::-1][:5]
    labels = [str(classes_[i]) for i in idx]
    scores = [float(proba[i]) for i in idx]
    short_term_text = f"Focus on developing strengths for {labels[0]} and {labels[1]} based on your current profile."
    long_term_text = f"Consider gaining experience leading toward {labels[0]} / {labels[2]} over the next 3–5 years."
    return render_template('results.html',
                           labels=labels, scores=scores,
                           top_labels=labels[:3],
                           short_term_text=short_term_text,
                           long_term_text=long_term_text)

# Static file passthrough for local dev
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory(os.path.join(os.path.dirname(__file__), '..', 'static'), path)

# Vercel requires 'app' to be exported
# For local run: `python api/index.py`
if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
