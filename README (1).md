# Smart Complaint Analyzer for Colleges

A web app where students submit complaints and an NLP/ML pipeline analyses each one automatically: it finds the category, sentiment, priority, and duplicates, and routes the complaint to the right department. Admins track and resolve complaints from a dashboard.

Built with Python, Streamlit, scikit-learn, NLTK, and SQLite.

## Features

**Students**
- Register and log in
- Write a complaint and instantly see its category, priority, and the department it was sent to
- Track their own complaints, status, admin remarks, and resolution time

**Admins**
- Log in as the super admin (sees everything) or a department admin (sees only their department)
- View complaints filtered by department, priority, and status. High priority comes first
- Update status (Submitted, In Progress, Resolved) with remarks. Resolving records a `resolved_at` time, and every change is saved in a history table
- Dashboard with totals, charts, average time to resolve, and recurring issues
- Super admin can reassign complaints, for example those sent to manual review

## How the AI pipeline works

```
Complaint text
  -> text cleaning (lowercase, remove noise, stopwords, lemmatization)
  -> category classification (TF-IDF + Logistic Regression, 13 categories)
  -> sentiment analysis (VADER + complaint-specific words and rules)
  -> duplicate detection (TF-IDF cosine similarity against open complaints)
  -> priority (High / Medium / Low, from safety keywords, sentiment, deadlines, repeats)
  -> department routing (category map, safety complaints go to Student Welfare,
     low-confidence complaints go to manual review)
```

Categories: Academic Support and Resources, Food and Cantines, Financial Support, Online learning, Career opportunities, Athletics and sports, Housing and Transportation, Health and Well-being Support, Activities and Travelling, Student Affairs, Hostel or Infrastructure, Safety & Discipline, Library.

## Project files

| File | Purpose |
|---|---|
| `app.py` | The whole application (AI pipeline, database, student and admin pages) |
| `requirements.txt` | Python libraries needed |
| `college_complaints_dataset.csv` | Training data (1,014 complaints, 78 per category) |

The database file `complaints.db` is created automatically on first run.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the link shown in the terminal (usually http://localhost:8501). The first start takes a little time because the model trains from the CSV.

## Deploy on Streamlit Community Cloud (free)

1. Upload `app.py`, `requirements.txt`, and `college_complaints_dataset.csv` to a **public** GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose the repository, and set the main file to `app.py`.
4. Open **Advanced settings -> Secrets** and add your own admin password:
   ```
   ADMIN_PASSWORD = "choose-a-strong-password"
   ```
5. Click **Deploy**. The first start takes 2 to 4 minutes.

## Admin accounts

Admins are created automatically on first run. They all use the password from the `ADMIN_PASSWORD` secret. If no secret is set, the password is `admin123`, which is for local testing only.

| Admin ID | Access |
|---|---|
| `admin` | All departments, plus reassigning |
| `hostel`, `mess`, `academic`, `scholarship`, `transport`, `library`, `placement`, `sports`, `activities`, `affairs`, `welfare`, `discipline` | Their own department only |

Students register themselves from the Student page.

## Demo data

On first start the app adds 40 sample complaints (student ID `demo`) so the dashboard is not empty. Their statuses and resolution times are random. To turn this off, delete the demo-data block in the `setup_once()` function in `app.py`.

## Limitations

- **Data is not permanent on Streamlit Cloud.** The database resets when the app restarts or is redeployed. For real use, move to a hosted database (such as Supabase or Turso).
- **The training data is synthetic.** It was written to cover the 13 categories, so accuracy on real student complaints will be lower than the roughly 96% measured on the test split. Adding real complaints from your college is the best way to improve it.
- **Sentiment and priority are rule-based.** They only know the words and patterns in the lists, and they can miss misspelled or unusual wording.
- **There is no "Others" category.** A complaint outside the 13 categories is placed in the closest one, or sent to manual review if the model is unsure.
- **Duplicate detection** links each complaint to its single closest match, so one problem reported many times can split into several groups.
- Anyone with the app link can register as a student, so share it only with people you trust.

## Possible improvements

- Persistent database (Supabase, Turso, or PostgreSQL)
- Email notifications when a complaint status changes
- Train on real complaints collected from the college
- A transformer-based sentiment model (such as a Hugging Face model)
- Password reset and admin management page
