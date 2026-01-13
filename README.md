#  Student Dropout Risk Prediction System (Flask + ML + Gemini AI)
An **AI-powered student dropout risk prediction platform** built using **Flask**, **Machine Learning**, **SQLite**, and **Google Gemini AI**.
The system helps **teachers, counsellors, and administrators** identify at-risk students early and take preventive actions.
##  Features
###  Machine Learning–Based Prediction
* Uses a **trained Random Forest model** (`dropout_rf.pkl`)
* Predicts:

  * **Dropout probability**
  * **Binary risk status (Yes / No)**
* Includes **feature-level impact explanations**

###  AI Counselling Insights (Gemini AI)

* Integrates **Google Gemini 1.5 Flash**
* Generates **actionable recommendations** for:

  * Students
  * Parents
  * Teachers
* Automatically falls back to **rule-based counselling** if Gemini is unavailable

### 🗂 Student Management

* View all students
* Detailed student profile
* Attendance, marks, income, distance, fee delay & engagement tracking

### 📝 Counsellor Notes

* Add notes for individual students
* Notes are **timestamped & author-tagged**

### 🔐 Secure Authentication

* Token-based authentication (UUID)
* Password hashing using **SHA-256**
* Session expiry support

### 📤 Bulk Upload

* Upload student records via **CSV**
* Auto-updates existing students

---

## 🧱 Tech Stack

| Layer       | Technology             |
| ----------- | ---------------------- |
| Backend     | Flask (Python)         |
| Database    | SQLite                 |
| ML Model    | Random Forest (Pickle) |
| AI Insights | Google Gemini API      |
| Frontend    | HTML (Jinja Templates) |
| Auth        | Token-based (UUID)     |

---

## 📁 Project Structure

```
├── app.py
├── app_data.db
├── students_500_realistic.csv
├── models/
│   ├── dropout_rf.pkl
│   └── accuracy.txt
├── templates/
│   ├── index.html
│   └── prediction.html
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Create Virtual Environment (Optional)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install flask numpy pandas google-generativeai
```

---

## 🔑 Gemini AI Configuration (Optional but Recommended)

Set your Gemini API key as an environment variable:

```bash
set GEMINI_API_KEY=your_api_key_here   # Windows
export GEMINI_API_KEY=your_api_key_here # Linux/Mac
```

> ⚠️ If not set, the app **automatically uses rule-based counselling insights**

---

## ▶️ Run the Application

```bash
python app.py
```

Server will start at:

```
http://127.0.0.1:5000
```

---

## 🔐 Default Login Credentials

```
Username: counsellor
Password: password123
```

> ⚠️ Change credentials before production use

---

## 📡 API Endpoints

### 🔑 Authentication

```
POST /api/login
```

### 👨‍🎓 Students

```
GET /api/students
GET /api/students/<student_id>
```

### 📊 Prediction

```
GET /api/predict/<student_id>
```

### 📝 Notes

```
POST /api/notes
Headers: Authorization: Bearer <token>
```

### 📤 Upload CSV

```
POST /api/upload
Headers: Authorization: Bearer <token>


## 📈 Prediction Output Includes

* Dropout probability
* Risk reasons
* Feature impact explanation
* Model accuracy
* AI-generated counselling guidance

---

## 🧪 Dataset

* `students_500_realistic.csv`
* Automatically seeded on first run
* Replaceable with real institutional data


## ⚠️ Disclaimer

This system is **decision-support software**, not a replacement for human judgement.
Predictions should be used responsibly alongside counsellor expertise.
