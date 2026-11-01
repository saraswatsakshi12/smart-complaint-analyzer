import os, re, hmac, hashlib, random, sqlite3, threading
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Smart Complaint Analyzer", page_icon="📝", layout="wide")

DB_PATH = "complaints.db"
DATASET = "college_complaints_dataset.csv"
STATUSES = ["Submitted", "In Progress", "Resolved"]
PRIORITIES = ["High", "Medium", "Low"]

HIGH_RISK = r"\b(suicide|self harm|har+as+\w*|ragging|assault\w*|molest\w*|threat\w*|injur\w*|fire|electric shock|food poisoning|fainted|unsafe)\b"
WELFARE_RISK = r"\b(suicid\w*|self[- ]?harm|har+as+\w*|ragging|bully\w*|bullied|stalk\w*|assault\w*|molest\w*)\b"
TIME_WORDS = r"\b(urgent|today|tomorrow|deadline|last date|immediately|asap)\b"
NEG_PHRASES = ["not working", "not coming", "not available", "not responding", "not received",
               "not credited", "not issued", "not clean", "no response", "no electricity",
               "no water", "never comes", "without any reason", "without informing", "without notice"]
NEG_PATTERN = r"\b(?:not|no|never)\b\s+(?:\w+\s+){0,2}(?:work|working|complet|respond|help|receiv|issu|clean|come|coming|available|credit)"
LEXICON = {"broken": -2.5, "dirty": -2.5, "late": -1.5, "delay": -1.8, "delayed": -1.8, "slow": -1.5,
           "rude": -2.5, "poor": -2.2, "worst": -3, "unsafe": -2.5, "overcrowded": -2, "outdated": -1.8,
           "undercooked": -2, "tasteless": -2, "stale": -2, "insects": -2.5, "unfair": -2.5, "ignored": -2,
           "unacceptable": -3, "harassment": -3, "struggling": -2, "frustrating": -2.5, "leaking": -1.8,
           "unhygienic": -2.5, "cancelled": -1.5, "wrongly": -1.8, "lack": -2, "difficult": -1.8,
           "difficulty": -1.8, "inadequate": -2.2, "unable": -1.8, "problem": -1.8, "problems": -1.8,
           "issue": -1.5, "issues": -1.5}
DEPARTMENT_MAP = {
    "Hostel or Infrastructure": "Hostel & Maintenance Office",
    "Financial Support": "Scholarship Section",
    "Academic Support and Resources": "Academic Block",
    "Online learning": "Academic Block",
    "Food and Cantines": "Mess Committee",
    "Housing and Transportation": "Transport Department",
    "Library": "Library",
    "Safety & Discipline": "Student Welfare / Discipline Committee",
    "Career opportunities": "Placement Cell",
    "Athletics and sports": "Sports Department",
    "Health and Well-being Support": "Student Welfare / Counselling Cell",
    "Activities and Travelling": "Student Activities Office",
    "Student Affairs": "Student Affairs Office",
}
ADMINS = [("admin", "ALL"), ("hostel", "Hostel & Maintenance Office"), ("mess", "Mess Committee"),
          ("academic", "Academic Block"), ("scholarship", "Scholarship Section"),
          ("transport", "Transport Department"), ("library", "Library"), ("placement", "Placement Cell"),
          ("sports", "Sports Department"), ("activities", "Student Activities Office"),
          ("affairs", "Student Affairs Office"), ("welfare", "Student Welfare / Counselling Cell"),
          ("discipline", "Student Welfare / Discipline Committee")]


# ---------------------------------------------------------------- AI models (built once, then cached)
@st.cache_resource
def load_resources():
    import nltk
    for pkg in ["stopwords", "wordnet", "omw-1.4", "vader_lexicon"]:
        nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    keep = {"not", "no", "nor", "never", "against", "very", "too", "down", "off"}
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(LEXICON)
    R = {"stop": set(stopwords.words("english")) - keep, "lem": WordNetLemmatizer(), "sia": sia}

    data = pd.read_csv(DATASET).rename(columns={"Reports": "text", "Genre": "category"})
    data = data.drop_duplicates("text").reset_index(drop=True)
    cleaned = data["text"].apply(lambda t: _clean(t, R))
    R["vectorizer"] = TfidfVectorizer(ngram_range=(1, 2))
    X = R["vectorizer"].fit_transform(cleaned)
    R["model"] = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, data["category"])
    R["texts"] = data["text"].tolist()
    return R


def _clean(text, R):
    text = re.sub(r"http\S+|www\.\S+", " ", text.lower())
    text = re.sub(r"[^a-z\s]", " ", text)
    out = []
    for w in text.split():
        if w in R["stop"] or len(w) < 2:
            continue
        out.append(R["lem"].lemmatize(R["lem"].lemmatize(w, "v"), "n"))
    return " ".join(out)


R = load_resources()


def clean_text(text):
    return _clean(text, R)


def get_sentiment(text):
    score = R["sia"].polarity_scores(text)["compound"]
    t = text.lower()
    hits = sum(1 for p in NEG_PHRASES if p in t) + len(re.findall(NEG_PATTERN, t))
    score = max(-1.0, score - 0.4 * hits)
    label = ("Very Negative" if score <= -0.5 else "Negative" if score <= -0.05
             else "Neutral" if score < 0.05 else "Positive")
    return label, round(score, 2)


def get_priority(text, category, sentiment_score, duplicate_count=0):
    t = text.lower()
    if re.search(HIGH_RISK, t):
        return "High"
    points = 0
    if sentiment_score <= -0.5: points += 2
    elif sentiment_score <= -0.05: points += 1
    if re.search(TIME_WORDS, t): points += 2
    if category == "Financial Support": points += 1
    if duplicate_count >= 2: points += 1
    return "High" if points >= 4 else "Medium" if points >= 2 else "Low"


def get_department(category, text, confidence):
    if re.search(WELFARE_RISK, text.lower()):
        return "Student Welfare / Counselling Cell"
    if confidence < 0.15:
        return "Admin Office (manual review)"
    return DEPARTMENT_MAP.get(category, "Admin Office (manual review)")


# ---------------------------------------------------------------- database
LOCK = threading.Lock()


@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS students (student_id TEXT PRIMARY KEY, name TEXT, password TEXT);
    CREATE TABLE IF NOT EXISTS admins (admin_id TEXT PRIMARY KEY, name TEXT, department TEXT, password TEXT);
    CREATE TABLE IF NOT EXISTS complaints (
        complaint_id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, complaint_text TEXT,
        created_at TEXT, status TEXT DEFAULT 'Submitted', cleaned_text TEXT, category TEXT,
        confidence REAL, sentiment TEXT, sentiment_score REAL, priority TEXT, department TEXT,
        duplicate_of INTEGER, remarks TEXT, resolved_at TEXT);
    CREATE TABLE IF NOT EXISTS status_history (
        history_id INTEGER PRIMARY KEY AUTOINCREMENT, complaint_id INTEGER, old_status TEXT,
        new_status TEXT, remarks TEXT, changed_by TEXT, changed_at TEXT);
    """)
    return conn


conn = get_conn()


def db(sql, params=(), write=False):
    with LOCK:
        cur = conn.execute(sql, params)
        if write:
            conn.commit()
            return cur.lastrowid
        return cur.fetchall()


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def hash_pw(password, salt=None):
    salt = salt or os.urandom(16).hex()
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 100_000).hex()
    return salt + "$" + h


def check_pw(password, stored):
    return hmac.compare_digest(hash_pw(password, stored.split("$")[0]), stored)


def find_duplicates(new_cleaned, exclude_id, threshold=0.6):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    rows = db("SELECT complaint_id, cleaned_text FROM complaints WHERE status != 'Resolved' "
              "AND cleaned_text IS NOT NULL AND complaint_id != ?", (exclude_id,))
    if not rows:
        return []
    ids, texts = zip(*rows)
    vec = TfidfVectorizer(ngram_range=(1, 2)).fit(list(texts) + [new_cleaned])
    sims = cosine_similarity(vec.transform([new_cleaned]), vec.transform(texts))[0]
    matches = [(i, float(s)) for i, s in zip(ids, sims) if s >= threshold]
    return sorted(matches, key=lambda x: x[1], reverse=True)


def process_complaint(cid):
    text = db("SELECT complaint_text FROM complaints WHERE complaint_id = ?", (cid,))[0][0]
    cleaned = clean_text(text)
    probs = R["model"].predict_proba(R["vectorizer"].transform([cleaned]))[0]
    category, confidence = R["model"].classes_[probs.argmax()], round(float(probs.max()), 2)
    label, score = get_sentiment(text)
    matches = find_duplicates(cleaned, cid)
    priority = get_priority(text, category, score, len(matches))
    department = get_department(category, text, confidence)
    db("UPDATE complaints SET cleaned_text=?, category=?, confidence=?, sentiment=?, sentiment_score=?, "
       "priority=?, department=?, duplicate_of=? WHERE complaint_id=?",
       (cleaned, category, confidence, label, score, priority, department,
        matches[0][0] if matches else None, cid), write=True)
    return category, priority, department


def setup_once():
    try:
        admin_pw = st.secrets.get("ADMIN_PASSWORD", "admin123")
    except Exception:
        admin_pw = "admin123"
    if not db("SELECT 1 FROM admins LIMIT 1"):
        for aid, dept in ADMINS:
            db("INSERT INTO admins VALUES (?,?,?,?)",
               (aid, "Super Admin" if dept == "ALL" else aid.capitalize() + " Desk", dept, hash_pw(admin_pw)), write=True)
    if not db("SELECT 1 FROM complaints LIMIT 1"):          # demo data so the dashboard is not empty
        random.seed(1)
        for text in random.sample(R["texts"], 40):
            created = datetime.now() - timedelta(days=random.randint(0, 20), hours=random.randint(0, 23))
            cid = db("INSERT INTO complaints (student_id, complaint_text, created_at) VALUES ('demo', ?, ?)",
                     (text, created.strftime("%Y-%m-%d %H:%M:%S")), write=True)
            process_complaint(cid)
            r = random.random()
            if r < 0.4:
                done = min(created + timedelta(hours=random.randint(2, 120)), datetime.now())
                db("UPDATE complaints SET status='Resolved', remarks='Resolved (demo data)', resolved_at=? "
                   "WHERE complaint_id=?", (done.strftime("%Y-%m-%d %H:%M:%S"), cid), write=True)
            elif r < 0.6:
                db("UPDATE complaints SET status='In Progress', remarks='Being checked (demo data)' "
                   "WHERE complaint_id=?", (cid,), write=True)


@st.cache_resource
def init_data():
    setup_once()
    return True


init_data()


# ---------------------------------------------------------------- student pages
def student_page():
    user = st.session_state.get("student")
    if not user:
        tab_login, tab_reg = st.tabs(["Login", "Register"])
        with tab_login:
            with st.form("student_login"):
                sid = st.text_input("Student ID", key="sl_id").strip()
                pw = st.text_input("Password", type="password", key="sl_pw")
                if st.form_submit_button("Login"):
                    row = db("SELECT student_id, name, password FROM students WHERE student_id=?", (sid,))
                    if row and check_pw(pw, row[0][2]):
                        st.session_state["student"] = (row[0][0], row[0][1])
                        st.rerun()
                    st.error("Wrong student ID or password.")
        with tab_reg:
            with st.form("student_register"):
                sid = st.text_input("Student ID", key="sr_id").strip()
                name = st.text_input("Name", key="sr_name").strip()
                pw = st.text_input("Password (min 6 characters)", type="password", key="sr_pw")
                if st.form_submit_button("Register"):
                    if not sid or not name or len(pw) < 6:
                        st.error("Fill every field. Password needs at least 6 characters.")
                    elif db("SELECT 1 FROM students WHERE student_id=?", (sid,)):
                        st.error("This student ID is already registered.")
                    else:
                        db("INSERT INTO students VALUES (?,?,?)", (sid, name, hash_pw(pw)), write=True)
                        st.success("Registered! Now use the Login tab.")
        return

    st.subheader("Welcome, " + user[1])
    with st.form("new_complaint", clear_on_submit=True):
        text = st.text_area("Write your complaint", height=140, key="complaint_text")
        send = st.form_submit_button("Submit complaint")
    if send:
        text = text.strip()
        if len(text) < 20:
            st.error("Please write at least 20 characters.")
        else:
            cid = db("INSERT INTO complaints (student_id, complaint_text, created_at) VALUES (?,?,?)",
                     (user[0], text, now_str()), write=True)
            category, priority, dept = process_complaint(cid)
            st.success(f"Complaint #{cid} submitted. Category: {category} | Priority: {priority} | Sent to: {dept}")

    st.markdown("#### My complaints")
    rows = db("SELECT complaint_id, created_at, complaint_text, status, remarks, resolved_at "
              "FROM complaints WHERE student_id=? ORDER BY complaint_id DESC", (user[0],))
    if rows:
        st.dataframe(pd.DataFrame(rows, columns=["ID", "Date", "Complaint", "Status", "Remarks", "Resolved at"]),
                     width="stretch", hide_index=True)
    else:
        st.info("You have not written any complaints yet.")


# ---------------------------------------------------------------- admin pages
def get_complaints(admin_dept, department=None, priority=None, status=None):
    sql = ("SELECT complaint_id, created_at, category, priority, status, department, duplicate_of, "
           "complaint_text FROM complaints WHERE 1=1")
    params = []
    dept = admin_dept if admin_dept != "ALL" else department
    for col, val in (("department", dept), ("priority", priority), ("status", status)):
        if val:
            sql += f" AND {col} = ?"
            params.append(val)
    sql += " ORDER BY CASE priority WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END, created_at"
    return db(sql, params)


def show_dashboard(admin):
    sql = ("SELECT complaint_id, created_at, category, sentiment, priority, status, department, duplicate_of, "
           "resolved_at FROM complaints")
    params = ()
    if admin[2] != "ALL":
        sql, params = sql + " WHERE department = ?", (admin[2],)
    data = pd.DataFrame(db(sql, params), columns=["complaint_id", "created_at", "category", "sentiment", "priority",
                                                  "status", "department", "duplicate_of", "resolved_at"])
    if data.empty:
        st.info("No complaints to show yet.")
        return
    data["hours"] = (pd.to_datetime(data["resolved_at"]) - pd.to_datetime(data["created_at"])).dt.total_seconds() / 3600
    is_dup = data["duplicate_of"].notna() & (data["duplicate_of"] < data["complaint_id"])
    resolved = int((data["status"] == "Resolved").sum())
    avg = data["hours"].mean()
    c = st.columns(5)
    c[0].metric("Total", len(data))
    c[1].metric("Resolved", f"{resolved} ({round(100 * resolved / len(data))}%)")
    c[2].metric("High priority, open", int(((data["priority"] == "High") & (data["status"] != "Resolved")).sum()))
    c[3].metric("Duplicates", int(is_dup.sum()))
    c[4].metric("Avg. hours to resolve", "-" if pd.isna(avg) else round(avg, 1))
    a, b = st.columns(2)
    a.markdown("**By category**"); a.bar_chart(data["category"].value_counts())
    b.markdown("**By priority**"); b.bar_chart(data["priority"].value_counts().reindex(PRIORITIES).fillna(0))
    a, b = st.columns(2)
    a.markdown("**By status**"); a.bar_chart(data["status"].value_counts().reindex(STATUSES).fillna(0))
    if admin[2] == "ALL":
        b.markdown("**Open complaints by department**")
        b.bar_chart(data[data["status"] != "Resolved"]["department"].value_counts())
    else:
        b.markdown("**By sentiment**"); b.bar_chart(data["sentiment"].value_counts())
    repeats = data[is_dup].groupby("duplicate_of").size().sort_values(ascending=False).head(5)
    if len(repeats):
        st.markdown("**Recurring issues**")
        for oid, n in repeats.items():
            txt = db("SELECT complaint_text FROM complaints WHERE complaint_id=?", (int(oid),))[0][0]
            st.write(f"ID {int(oid)}: {txt[:80]} - reported {n + 1} times")


def update_tab(admin, rows):
    if not rows:
        st.info("No complaints available.")
        return
    options = {f"#{r[0]} | {r[3]} | {r[4]} | {r[7][:50]}": r[0] for r in rows}
    cid = options[st.selectbox("Choose a complaint", list(options), key="upd_pick")]
    text, dept, old, old_rem, prio = db("SELECT complaint_text, department, status, remarks, priority "
                                        "FROM complaints WHERE complaint_id=?", (cid,))[0]
    st.write(f"**Complaint:** {text}")
    st.caption(f"Department: {dept} | Priority: {prio} | Status: {old}")
    hist = db("SELECT changed_at, changed_by, old_status, new_status, remarks FROM status_history "
              "WHERE complaint_id=? ORDER BY history_id", (cid,))
    if hist:
        st.dataframe(pd.DataFrame(hist, columns=["When", "By", "From", "To", "Remarks"]),
                     width="stretch", hide_index=True)
    with st.form("update_form"):
        new = st.selectbox("New status", STATUSES, index=STATUSES.index(old), key="upd_status")
        remarks = st.text_input("Remarks (required when resolving)", key="upd_remarks").strip()
        if st.form_submit_button("Update"):
            if new == old:
                st.warning("Status is unchanged.")
            elif new == "Resolved" and not remarks:
                st.error("Please write remarks explaining how it was resolved.")
            else:
                now = now_str()
                db("UPDATE complaints SET status=?, remarks=?, resolved_at=? WHERE complaint_id=?",
                   (new, remarks or old_rem, now if new == "Resolved" else None, cid), write=True)
                db("INSERT INTO status_history (complaint_id, old_status, new_status, remarks, changed_by, changed_at) "
                   "VALUES (?,?,?,?,?,?)", (cid, old, new, remarks, admin[0], now), write=True)
                st.success(f"Complaint {cid} is now {new}. Reload the list to see it.")


def reassign_tab(admin):
    cid = st.number_input("Complaint ID", min_value=1, step=1, key="re_id")
    depts = sorted({a[1] for a in ADMINS if a[1] != "ALL"})
    new_dept = st.selectbox("Send to department", depts, key="re_dept")
    if st.button("Reassign", key="re_btn"):
        row = db("SELECT department, status FROM complaints WHERE complaint_id=?", (int(cid),))
        if not row:
            st.error("No complaint with this ID.")
        else:
            db("UPDATE complaints SET department=? WHERE complaint_id=?", (new_dept, int(cid)), write=True)
            db("INSERT INTO status_history (complaint_id, old_status, new_status, remarks, changed_by, changed_at) "
               "VALUES (?,?,?,?,?,?)", (int(cid), row[0][1], row[0][1],
                                        f"Reassigned from {row[0][0]} to {new_dept}", admin[0], now_str()), write=True)
            st.success(f"Complaint {int(cid)} sent to {new_dept}.")


def admin_page():
    admin = st.session_state.get("admin")
    if not admin:
        with st.form("admin_login"):
            aid = st.text_input("Admin ID", key="al_id").strip()
            pw = st.text_input("Password", type="password", key="al_pw")
            if st.form_submit_button("Login"):
                row = db("SELECT admin_id, name, department, password FROM admins WHERE admin_id=?", (aid,))
                if row and check_pw(pw, row[0][3]):
                    st.session_state["admin"] = row[0][:3]
                    st.rerun()
                st.error("Wrong admin ID or password.")
        return

    st.subheader(f"{admin[1]} ({admin[2]})")
    tabs = ["Complaints", "Update status", "Dashboard"] + (["Reassign"] if admin[2] == "ALL" else [])
    t = st.tabs(tabs)
    with t[0]:
        f = st.columns(3)
        dept = None
        if admin[2] == "ALL":
            depts = [r[0] for r in db("SELECT DISTINCT department FROM complaints WHERE department IS NOT NULL")]
            dept = f[0].selectbox("Department", [None] + sorted(depts), format_func=lambda x: "All" if x is None else x, key="f_dept")
        prio = f[1].selectbox("Priority", [None] + PRIORITIES, format_func=lambda x: "All" if x is None else x, key="f_prio")
        stat = f[2].selectbox("Status", [None] + STATUSES, format_func=lambda x: "All" if x is None else x, key="f_stat")
        rows = get_complaints(admin[2], dept, prio, stat)
        st.caption(f"{len(rows)} complaint(s)")
        if rows:
            st.dataframe(pd.DataFrame(rows, columns=["ID", "Created", "Category", "Priority", "Status",
                                                     "Department", "Dup of", "Complaint"]),
                         width="stretch", hide_index=True)
    with t[1]:
        update_tab(admin, get_complaints(admin[2]))
    with t[2]:
        show_dashboard(admin)
    if admin[2] == "ALL":
        with t[3]:
            reassign_tab(admin)


# ---------------------------------------------------------------- main
st.title("📝 Smart Complaint Analyzer")
role = st.sidebar.radio("I am a", ["Student", "Admin"], key="role")
if st.session_state.get("student") and role == "Student" or st.session_state.get("admin") and role == "Admin":
    if st.sidebar.button("Logout"):
        st.session_state.pop("student" if role == "Student" else "admin", None)
        st.rerun()
st.sidebar.caption("Demo project. Includes 40 sample complaints marked student ID 'demo'.")
student_page() if role == "Student" else admin_page()
