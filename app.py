from flask import Flask, jsonify, request, render_template, g
import sqlite3, os, pickle, datetime, uuid, hashlib
import numpy as np

# ✅ Gemini (AI Insights)
import google.generativeai as genai

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, 'app_data.db')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'dropout_rf.pkl')

app = Flask(__name__, template_folder='templates')

# 🔑 Gemini API Key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY"Place key here")
if GEMINI_API_KEY != "Place key here":
    genai.configure(api_key=GEMINI_API_KEY)

# Load model & meta
with open(MODEL_PATH, 'rb') as f:
    mdl = pickle.load(f)
model = mdl['model']
meta = mdl['meta']
FEATURES = meta['features']
FEATURE_MEANS = meta['feature_means']
FEATURE_IMPS = meta['feature_importances']

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS students (
        student_id TEXT PRIMARY KEY, name TEXT, class INTEGER, village TEXT,
        attendance INTEGER, result_pct REAL, assignments_submitted INTEGER,
        engagement_score INTEGER, family_income INTEGER, distance_km REAL,
        fee_delay INTEGER, risk_score REAL, dropout INTEGER
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, note TEXT, created_at TEXT, author TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password_hash TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS tokens (
        token TEXT PRIMARY KEY, user_id INTEGER, expires_at TEXT
    )""")
    conn.commit(); conn.close()

def seed_students_from_csv():
    import pandas as pd
    df = pd.read_csv(os.path.join(BASE_DIR, 'students_500_realistic.csv'))
    conn = sqlite3.connect(DB_PATH)
    df.to_sql('students', conn, if_exists='replace', index=False)
    conn.close()

def create_default_user():
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    try:
        cur.execute("SELECT 1 FROM users WHERE username='counsellor'")
        if cur.fetchone() is None:
            username='counsellor'; password='password123'
            salt = username[::-1]
            ph = hashlib.sha256((salt+password).encode()).hexdigest()
            cur.execute('INSERT INTO users(username,password_hash) VALUES (?,?)', (username,ph))
            conn.commit()
    except:
        pass
    conn.close()

@app.teardown_appcontext
def close_db(exc):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# -------------------- FRONTEND ROUTES --------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction')
def prediction_page():
    return render_template('prediction.html')

# -------------------- API ROUTES --------------------

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json(force=True)
    username = data.get('username'); password = data.get('password')
    if not username or not password:
        return jsonify({'error':'username and password required'}), 400
    db = get_db(); cur = db.cursor()
    cur.execute('SELECT id, password_hash FROM users WHERE username=?', (username,))
    r = cur.fetchone()
    if not r:
        return jsonify({'error':'invalid credentials'}), 401
    user_id = r['id']; stored = r['password_hash']
    salt = username[::-1]
    if hashlib.sha256((salt+password).encode()).hexdigest() != stored:
        return jsonify({'error':'invalid credentials'}), 401
    token = str(uuid.uuid4())
    expires = (datetime.datetime.utcnow() + datetime.timedelta(hours=6)).isoformat()
    cur.execute('INSERT OR REPLACE INTO tokens(token,user_id,expires_at) VALUES (?,?,?)', (token, user_id, expires))
    get_db().commit()
    return jsonify({'token': token, 'expires_at': expires})

def verify_token(auth_header):
    if not auth_header: return None
    parts = auth_header.split()
    if len(parts) != 2: return None
    token = parts[1]
    db = get_db(); cur = db.cursor()
    cur.execute('SELECT token, user_id, expires_at FROM tokens WHERE token=?', (token,))
    r = cur.fetchone()
    if not r: return None
    try:
        if datetime.datetime.fromisoformat(r['expires_at']) < datetime.datetime.utcnow():
            return None
    except:
        return None
    return r['user_id']

@app.route('/api/students', methods=['GET'])
def api_students():
    db = get_db(); cur = db.cursor()
    cur.execute('SELECT student_id, name, class, attendance, result_pct, risk_score, dropout, village FROM students')
    rows = cur.fetchall()
    return jsonify([dict(x) for x in rows])

@app.route('/api/students/<student_id>', methods=['GET'])
def api_student(student_id):
    db = get_db(); cur = db.cursor()
    cur.execute('SELECT * FROM students WHERE student_id=?', (student_id,))
    r = cur.fetchone()
    if not r: return jsonify({'error':'not found'}), 404
    student = dict(r)
    cur.execute('SELECT note, created_at, author FROM notes WHERE student_id=? ORDER BY created_at DESC', (student_id,))
    notes = [dict(x) for x in cur.fetchall()]
    return jsonify({'student': student, 'notes': notes})

@app.route('/api/predict/<student_id>', methods=['GET'])
def api_predict(student_id):
    db = get_db(); cur = db.cursor()
    cur.execute('SELECT * FROM students WHERE student_id=?', (student_id,))
    r = cur.fetchone()
    if not r: return jsonify({'error':'not found'}), 404
    s = dict(r)

    # prediction
    X = np.array([s.get(f,0) for f in FEATURES]).reshape(1, -1)
    prob = float(model.predict_proba(X)[0][1]) if hasattr(model, 'predict_proba') else float(model.predict(X)[0])
    pred = int(prob >= 0.5)

    # rule-based reasons
    reasons = []
    if s.get('attendance',100) < 70: reasons.append('Low attendance')
    if s.get('result_pct',100) < 40: reasons.append('Low exam percentage')
    if s.get('family_income',100000) < 5000: reasons.append('Low family income (financial stress)')
    if s.get('distance_km',0) > 10: reasons.append('Long travel distance (>10 km)')
    if s.get('fee_delay',0) == 1: reasons.append('Fee/payment delays')
    if not reasons: reasons.append('Combined factors increase risk')

    # feature impacts
    impacts = []
    for f in FEATURES:
        val = float(s.get(f,0))
        mean = float(FEATURE_MEANS.get(f,0))
        imp = float(FEATURE_IMPS.get(f,0))
        impact = (val - mean) * imp
        impacts.append({'feature': f, 'value': val, 'impact': round(impact,4)})

    # accuracy
    try:
        with open(os.path.join(BASE_DIR,'models','accuracy.txt'),'r') as f:
            acc = float(f.read().strip())
    except:
        acc = 0.0

    # AI Insights using Gemini
    ai_insight = ""
    if GEMINI_API_KEY != "key here":
        prompt = f"""
        You are an education counsellor AI. Analyze the following student's data and risk factors.
        Student: {s}
        Predicted dropout risk: {"Yes" if pred else "No"}
        Probability: {round(prob*100,1)}%
        Risk reasons: {", ".join(reasons)}
        Provide clear protective measures, suggestions, and actionable insights for teachers, parents, and the student.
        """
        try:
            model_ai = genai.GenerativeModel("gemini-1.5-flash")
            resp = model_ai.generate_content(prompt)
            ai_insight = resp.text.strip()
        except Exception:
            pass   # fallback will be used below

    # ✅ Dynamic Role-Based Fallback if Gemini unavailable or failed
    if not ai_insight:
        measures = []

        if "Low attendance" in reasons:
            measures.append("For Student: Set daily study and class attendance targets.")
            measures.append("For Parents: Monitor attendance and encourage timely routine.")
            measures.append("For Teachers: Provide mentorship and track attendance regularly.")
        
        if "Low exam percentage" in reasons:
            measures.append("For Student: Revise weak subjects daily and join peer study groups.")
            measures.append("For Parents: Arrange a quiet study environment at home.")
            measures.append("For Teachers: Offer remedial classes and give constructive feedback.")
        
        if "Low family income (financial stress)" in reasons:
            measures.append("For Student: Focus on available free learning resources online.")
            measures.append("For Parents: Apply for scholarships or fee waivers if eligible.")
            measures.append("For Teachers: Connect family to school welfare schemes.")
        
        if "Long travel distance (>10 km)" in reasons:
            measures.append("For Student: Manage time effectively and avoid missing early classes.")
            measures.append("For Parents: Explore nearby accommodation or transport pooling.")
            measures.append("For Teachers: Allow flexible timings if travel causes delays.")
        
        if "Fee/payment delays" in reasons:
            measures.append("For Student: Stay motivated despite financial challenges.")
            measures.append("For Parents: Seek installment-based fee options.")
            measures.append("For Teachers: Offer counselling without discrimination.")

        if not measures:
            measures = [
                "For Student: Stay consistent with homework and participation.",
                "For Parents: Support learning with encouragement and monitoring.",
                "For Teachers: Hold monthly counselling sessions for guidance."
            ]

        ai_insight = "\n".join(measures[:5])

    return jsonify({
        'prediction': pred,
        'dropout_probability': round(prob,3),
        'reasons': reasons,
        'explanation': impacts,
        'model_accuracy': round(acc,4),
        'student': s,
        'ai_insight': ai_insight
    })

# -------------------- NOTES & UPLOAD --------------------

@app.route('/api/notes', methods=['POST'])
def api_notes():
    user_id = verify_token(request.headers.get('Authorization', None))
    if not user_id:
        return jsonify({'error':'unauthorized'}), 401
    data = request.get_json(force=True)
    sid = data.get('student_id'); note = data.get('note','').strip()
    if not sid or not note:
        return jsonify({'error':'student_id and note required'}), 400
    author = request.headers.get('X-User','unknown')
    db = get_db(); cur = db.cursor()
    cur.execute('INSERT INTO notes(student_id, note, created_at, author) VALUES (?,?,?,?)',
                (sid, note, datetime.datetime.utcnow().isoformat(), author))
    db.commit()
    return jsonify({'status':'ok'})

@app.route('/api/upload', methods=['POST'])
def api_upload():
    user_id = verify_token(request.headers.get('Authorization', None))
    if not user_id:
        return jsonify({'error':'unauthorized'}), 401
    if 'file' not in request.files:
        return jsonify({'error':'no file part'}), 400
    f = request.files['file']
    import pandas as pd
    df = pd.read_csv(f)
    if 'student_id' not in df.columns:
        return jsonify({'error':'student_id column required'}), 400
    db = get_db(); cur = db.cursor()
    for _, row in df.iterrows():
        cols = list(row.index)
        placeholders = ','.join(['?']*len(cols))
        sql = "REPLACE INTO students (" + ','.join(cols) + ") VALUES (" + placeholders + ")"
        cur.execute(sql, tuple(row.values))
    db.commit()
    return jsonify({'status':'ok'})

# -------------------- INIT --------------------

if __name__ == '__main__':
    init_db()
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute('SELECT count(*) as c FROM students'); c = cur.fetchone()[0]
    conn.close()
    if c == 0:
        seed_students_from_csv()
    create_default_user()
    app.run(host='0.0.0.0', port=5000, debug=True)
