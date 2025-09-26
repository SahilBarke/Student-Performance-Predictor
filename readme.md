# 🎓 Student Performance Predictor

A simple Machine Learning web app that predicts a student's performance index based on factors like study hours, sleep, previous scores, and more.

---

## 🚀 Features

- ✅ Real-time prediction of performance index
- ✅ Clean and simple **Streamlit UI**
- ✅ Logically constrained inputs (e.g., study hours + sleep ≤ 24)
- ✅ Includes dataset for reference and experimentation

**Inputs considered:**
- 📖 **Study hours per day**
- 😴 **Sleep hours per day**
- 📝 **Previous test scores**
- 📚 **Sample papers practiced**
- ⚽ **Extracurricular activities** (scale 0–10)
  
---

## 🛠️ Tech Stack

- **Python 3**
- **pandas**, **NumPy**, **scikit-learn**, **statsmodels**
- **Streamlit** (frontend UI)
- **pickle** (for model serialization)

---

##  📊 Workflow

1. **Data Preparation** – Cleaned and explored dataset (EDA)
2. **Model Building** – Trained regression model using scikit-learn
3. **Evaluation** – Tested model accuracy and fine-tuned hyperparameters
4. **Deployment** – Built Streamlit app for real-time predictions

---

📂 The dataset used to build and train the model is also included in this repository for reference and experimentation.

---

## ▶️ Run Locally

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/Student-Performance-Predictor.git
cd Student-Performance-Predictor
pip install -r requirements.txt
```
Run the app:
```bash
streamlit run app.py

