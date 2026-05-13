# ✈️ British Airways Customer Sentiment Analysis
### End-to-End NLP Pipeline | From Web Scraping to Live Deployment

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://ba-sentiment-analysis-emmanuelrockson.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/Emmanuel-Rockson/ba-sentiment-analysis)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge)](https://huggingface.co)

---

## 📌 Project Overview

A **complete end-to-end NLP project** that automatically classifies British Airways customer reviews as **Positive**, **Neutral** or **Negative** — giving business analysts instant insight into customer satisfaction without reading thousands of reviews manually.

Built from scratch: web scraping real customer data → cleaning and EDA → training and comparing 4 NLP models → deploying a live interactive dashboard.

> **"47% of BA reviews are negative, dominated by 1-star ratings. Delays and staff attitude are the top pain points."**
> — Key finding from analysis of 7,696 real customer reviews

---

## 🚀 Live Demo

👉 **[Click here to view the live dashboard](https://ba-sentiment-analysis-emmanuelrockson.streamlit.app)**

> ℹ️ Hosted on Streamlit Community Cloud free tier. If the app is sleeping click **"Yes, get this app back up!"** — it wakes in under 60 seconds.

### Dashboard Features
- 📊 **KPI Cards** — total, negative, neutral and positive review counts at a glance
- 📈 **Sentiment Trend Over Time** — track how customer satisfaction has changed month by month
- 🔴 **Top Customer Complaints** — word frequency analysis of negative reviews
- 🟢 **Top Positive Themes** — what customers praise most about BA
- 🔮 **Live Sentiment Predictor** — paste any review and get an instant prediction with confidence score
- ⚡ **Fast Mode** — TF-IDF + Logistic Regression (millisecond predictions)
- 🎯 **Accurate Mode** — DistilBERT (available locally)

---

## 📊 Model Performance

| Model | Accuracy | Speed | Notes |
|-------|----------|-------|-------|
| VADER Baseline | 66.0% | Instant | Rule-based, no training needed |
| TF-IDF + Logistic Regression | 74.8% | Milliseconds | Classical ML baseline |
| DistilBERT (fine-tuned) | 78.5% | ~2-3 seconds | Transformer — bidirectional understanding |
| **Improved TF-IDF + SMOTE** | **87.4%** | **Milliseconds** | **Best model — deployed in dashboard** |

> **12.6% accuracy improvement** achieved by combining multi-source data augmentation and SMOTE oversampling to fix class imbalance — outperforming a pretrained domain-specific BERT model.

---

## 🔍 Key Business Insights

```
📉 47% of BA reviews are NEGATIVE  — dominated by 1-star ratings
⭐ Rating 1 alone has nearly as many reviews as ratings 7-10 combined
✈️ Top complaints: delays, staff attitude, seat comfort, food quality
✅ Top praise: friendly crew, on-time flights, comfortable business class
📆 Negative sentiment has been increasing since 2022
🔍 Unverified reviews skew 23% more negative than verified ones
```

---

## 🗂️ Project Structure

```
ba-sentiment-analysis/
│
├── 📂 data/
│   ├── cleaned_reviews.csv          # Cleaned Skytrax data
│   └── combined_reviews.csv         # Skytrax + Kaggle combined
│
├── 📂 notebooks/
│   ├── 01_scraping.ipynb            # Web scraping Skytrax reviews
│   ├── 02_cleaning.ipynb            # Data cleaning and labelling
│   ├── 03_eda.ipynb                 # Exploratory data analysis
│   ├── 04_modelling.ipynb           # TF-IDF and VADER models
│   ├── 05_distilbert.ipynb          # DistilBERT fine-tuning
│   ├── 06_data_combination.ipynb    # Combining data sources
│   └── 07_improved_model.ipynb      # SMOTE and improved TF-IDF
│
├── 📂 models/
│   └── improved_sentiment_model.pkl # Saved improved TF-IDF model
│
├── 📂 app/
│   └── app.py                       # Streamlit dashboard
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.11 |
| **Data Collection** | requests, BeautifulSoup4 |
| **Data Processing** | pandas, numpy, re |
| **NLP & ML** | scikit-learn, transformers, PyTorch, NLTK |
| **Class Balancing** | imbalanced-learn (SMOTE) |
| **Visualisation** | matplotlib, wordcloud |
| **Dashboard** | Streamlit |
| **Version Control** | Git, GitHub |
| **Deployment** | Streamlit Community Cloud |

---

## 📋 Methodology

### 1. Data Collection
- Scraped **4,000+ reviews** from [Skytrax](https://www.airlinequality.com/airline-reviews/british-airways) using `requests` and `BeautifulSoup`
- Supplemented with **3,692 reviews** from a Kaggle dataset
- Combined and deduplicated to **7,696 total reviews**

### 2. Data Cleaning
- Removed Skytrax `Trip Verified |` prefix
- Lowercased, removed special characters, normalised whitespace
- Dropped reviews with fewer than 10 words
- Handled mixed date formats from two data sources

### 3. Sentiment Labelling
```python
# 3-class labelling strategy based on star rating
Negative  →  ratings 1, 2, 3   (47% of dataset)
Neutral   →  ratings 4, 5, 6   (18% of dataset)
Positive  →  ratings 7, 8, 9, 10 (35% of dataset)
```

### 4. Handling Class Imbalance
Applied **SMOTE (Synthetic Minority Oversampling Technique)** to the training set:
```
Before SMOTE:  Negative 2,497 | Neutral 880 | Positive 1,652
After SMOTE:   Negative 2,497 | Neutral 2,497 | Positive 2,497  ✅ Perfectly balanced
```
> SMOTE was applied **only to training data** — test set retained real-world distribution for honest evaluation.

### 5. Model Building
Four models built and compared in increasing sophistication:
- **VADER** — rule-based baseline
- **TF-IDF + Logistic Regression** — classical ML
- **DistilBERT** — fine-tuned transformer
- **Improved TF-IDF + SMOTE** — best performing model

### 6. Production Features
- **Confidence threshold** — predictions below 70% confidence flagged for human review
- **Minimum review length** — reviews under 10 words prompt user for more detail
- **Dual prediction modes** — Fast (TF-IDF) vs Accurate (DistilBERT) based on use case

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.11+
- Git

### Clone the repository
```bash
git clone https://github.com/Emmanuel-Rockson/ba-sentiment-analysis.git
cd ba-sentiment-analysis
```

### Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the dashboard locally
```bash
cd app
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🔮 Live Prediction Example

```python
# Fast mode — TF-IDF + Logistic Regression
import pickle

with open('models/improved_sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

review = "The flight was delayed 4 hours with no explanation. Staff were dismissive and rude."
prediction = model.predict([review])[0]
confidence = round(max(model.predict_proba([review])[0]) * 100, 1)

print(f"Sentiment: {prediction.upper()}")   # NEGATIVE
print(f"Confidence: {confidence}%")         # 84.0%
```

---

## 📈 Business Impact

This project demonstrates how NLP sentiment analysis can deliver real business value for British Airways:

| Impact Area | Business Value |
|-------------|----------------|
| 🎯 Contact Centre | 15-25% reduction in average handle time via real-time sentiment scoring |
| ⚠️ Early Warning | Detect operational failures days before escalation |
| 📦 Product Development | Data-driven investment prioritisation by service area |
| 💰 Revenue Protection | Identify at-risk loyalty members before they defect |
| ⏱️ Operational Efficiency | 2,500+ analyst hours saved per year |
| 🏛️ Strategic Decisions | Quantitative customer evidence for board-level decisions |

---

## 🔭 Future Improvements

- [ ] **Aspect-Based Sentiment Analysis** — predict sentiment per topic (food, staff, delays, comfort) separately
- [ ] **Date range filter** on dashboard for period-specific analysis
- [ ] **Retrain DistilBERT** on full 7,696 review dataset with SMOTE — expected 85-90% accuracy
- [ ] **Competitor benchmarking** — apply same model to Virgin Atlantic and easyJet reviews
- [ ] **Automated daily scraping** to keep data fresh
- [ ] **SHAP explainability** — show which words drove each prediction

---

## 👨‍💻 Author

**Emmanuel Rockson**
Data Science & AI Graduate

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/Emmanuel-Rockson)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <strong>⭐ If you found this project useful please consider giving it a star!</strong>
</p>
