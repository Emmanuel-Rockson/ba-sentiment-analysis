
# ============================================================
# BRITISH AIRWAYS SENTIMENT ANALYSIS DASHBOARD
# ============================================================

# import libraries
import streamlit as st 
import pandas as pd  # pandas for loading and manipulating the  data
import matplotlib.pyplot as plt # matplotlib for drawing charts
from collections import Counter # Counter counts how many times each word appears
import re # re is for cleaning text with regular expressions
import pickle # pickle loads our saved trained models
import os # os helps us work with file paths

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="British Airways Sentiment Dashboard",
    page_icon="✈️",
    layout="wide"         # wide layout uses the full screen width
)

# ============================================================
# LOAD DATA
# I used st.cache_data so the data only loads once
# Without this it would reload every time the user clicks anything
# ============================================================
@st.cache_data
def load_data():
    # Go up one folder from app/ to find the data/ folder
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_reviews.csv')
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

# ============================================================
# LOAD MODELS
# I used st.cache_resource for models — they are heavy objects
# that should only be loaded into memory once
# ============================================================
@st.cache_resource
def load_models():
    # Build an absolute path so it works regardless of where you run the app from
    base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'sentiment_model.pkl')
    
    with open(model_path, 'rb') as f:
        tfidf_model = pickle.load(f)
    return tfidf_model

# Load data and model
@st.cache_data
def load_data():
    # Build an absolute path so it works regardless of where you run the app from
    base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'cleaned_reviews.csv')
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

# Load data and model
df = load_data()
tfidf_model = load_models()

# ============================================================
# DASHBOARD HEADER
# ============================================================
st.title("✈️ British Airways — Customer Sentiment Dashboard")
st.markdown("Analysing customer reviews scraped from Skytrax to surface business insights.")

# Add a dividing line
st.markdown("---")

# ============================================================
# SECTION 1 — KPI CARDS
# st.columns splits the page into side by side sections
# ============================================================
st.subheader("📊 Overall Sentiment Overview")

# Count reviews in each sentiment group
total      = len(df)
negative   = len(df[df['sentiment'] == 'negative'])
neutral    = len(df[df['sentiment'] == 'neutral'])
positive   = len(df[df['sentiment'] == 'positive'])

# Calculate percentages
neg_pct = round((negative / total) * 100, 1)
neu_pct = round((neutral  / total) * 100, 1)
pos_pct = round((positive / total) * 100, 1)

# Create 4 columns for the KPI cards
col1, col2, col3, col4 = st.columns(4)

# Each metric shows the count and the percentage change (delta)
# delta_color='inverse' means red is bad and green is good
col1.metric(
    label="Total Reviews",
    value=f"{total:,}"
)
col2.metric(
    label="Negative Reviews",
    value=f"{negative:,}",
    delta=f"{neg_pct}% of total",
    delta_color="inverse"    # red colour because negative is bad
)
col3.metric(
    label="Neutral Reviews",
    value=f"{neutral:,}",
    delta=f"{neu_pct}% of total",
    delta_color="off"        # grey colour for neutral
)
col4.metric(
    label="Positive Reviews",
    value=f"{positive:,}",
    delta=f"{pos_pct}% of total",
    delta_color="normal"     # green colour because positive is good
)

st.markdown("---")

# ============================================================
# SECTION 2 — SENTIMENT TREND OVER TIME
# ============================================================
st.subheader("📈 Sentiment Trend Over Time")

# Group reviews by month and sentiment
# This shows whether BA is getting better or worse over time
df['year_month'] = df['date'].dt.to_period('M').astype(str)
trend = df.groupby(['year_month', 'sentiment']).size().unstack(fill_value=0)

# Plot the trend
fig, ax = plt.subplots(figsize=(12, 4))
if 'negative' in trend.columns:
    ax.plot(trend.index, trend['negative'], color='#E24B4A', label='Negative', linewidth=2)
if 'neutral' in trend.columns:
    ax.plot(trend.index, trend['neutral'],  color='#EF9F27', label='Neutral',  linewidth=2)
if 'positive' in trend.columns:
    ax.plot(trend.index, trend['positive'], color='#1D9E75', label='Positive', linewidth=2)

ax.set_xlabel('Month')
ax.set_ylabel('Number of Reviews')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Only show every 6th month label so x axis doesn't get crowded
tick_positions = range(0, len(trend.index), 6)
ax.set_xticks(list(tick_positions))
ax.set_xticklabels([trend.index[i] for i in tick_positions], rotation=45)

plt.tight_layout()

# st.pyplot renders a matplotlib chart on the dashboard
st.pyplot(fig)

st.markdown("---")

# ============================================================
# SECTION 3 — TOP CUSTOMER COMPLAINTS
# ============================================================
st.subheader("🔴 Top Customer Complaints")
st.markdown("Most frequent words appearing in negative reviews — these are BA's key pain points.")

# Filter negative reviews only
negative_text = ' '.join(df[df['sentiment'] == 'negative']['clean_text'].dropna())

# Split into individual words and count them
words = negative_text.split()

# Words to ignore — they appear everywhere but mean nothing on their own
stopwords = [
    'the','a','and','to','was','is','in','i','of','for','my','we',
    'on','it','had','with','at','flight','british','airways','ba',
    'not','but','no','be','were','have','our','this','that','they',
    'very','so','as','an','are','has','from','would','could','when',
    'there','their','which','been','will','one','just','also','more'
]

# Count words and filter out stopwords and short words
word_counts = Counter(w for w in words if w not in stopwords and len(w) > 3)

# Get the top 10 complaint words
top_complaints = dict(word_counts.most_common(10))

# Plot horizontal bar chart
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.barh(list(top_complaints.keys()), list(top_complaints.values()), color='#E24B4A')
ax2.set_xlabel('Frequency')
ax2.set_title('Top 10 Words in Negative Reviews')
ax2.invert_yaxis()   # most common word at the top
plt.tight_layout()
st.pyplot(fig2)

st.markdown("---")

# ============================================================
# SECTION 4 — TOP POSITIVE THEMES
# ============================================================
st.subheader("🟢 Top Positive Themes")
st.markdown("Most frequent words in positive reviews — these are BA's strengths.")

# Same process but for positive reviews
positive_text = ' '.join(df[df['sentiment'] == 'positive']['clean_text'].dropna())
pos_words     = positive_text.split()
pos_counts    = Counter(w for w in pos_words if w not in stopwords and len(w) > 3)
top_positive  = dict(pos_counts.most_common(10))

fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.barh(list(top_positive.keys()), list(top_positive.values()), color='#1D9E75')
ax3.set_xlabel('Frequency')
ax3.set_title('Top 10 Words in Positive Reviews')
ax3.invert_yaxis()
plt.tight_layout()
st.pyplot(fig3)

st.markdown("---")

# ============================================================
# SECTION 5 — LIVE PREDICTION
# ============================================================
st.subheader("🔮 Live Sentiment Predictor")
st.markdown("Paste any British Airways review below to instantly predict its sentiment.")

# Dropdown to choose which model to use
model_choice = st.selectbox(
    "Choose prediction mode:",
    ["⚡ Fast mode — TF-IDF + Logistic Regression", 
     "🎯 Accurate mode — DistilBERT"]
)

# Text box where the user pastes a review
user_review = st.text_area(
    "Paste a customer review here:",
    height=150,
    placeholder="e.g. The flight was delayed 3 hours and the staff were very rude..."
)

# When the user clicks the button
if st.button("Analyse Sentiment"):

    # Make sure they actually typed something
    if user_review.strip() == '':
        st.warning("Please enter a review to analyse.")

    else:
        # Clean the input text the same way we cleaned the training data
        clean_input = re.sub(r'[^a-zA-Z0-9\s]', '', user_review.lower())

        if "Fast mode" in model_choice:
            # Use TF-IDF model for instant prediction
            prediction = tfidf_model.predict([clean_input])[0]

            # Get the confidence score for each class
            probabilities  = tfidf_model.predict_proba([clean_input])[0]
            confidence     = round(max(probabilities) * 100, 1)

        else:
            # Placeholder for DistilBERT prediction
            # We will wire this up properly in the next step
            st.info("DistilBERT mode coming soon — use Fast mode for now.")
            prediction  = None
            confidence  = None

        # Display the result with colour coding
        if prediction == 'positive':
            st.success(f"Sentiment: POSITIVE 😊 (Confidence: {confidence}%)")
        elif prediction == 'negative':
            st.error(f"Sentiment: NEGATIVE 😞 (Confidence: {confidence}%)")
        elif prediction == 'neutral':
            st.warning(f"Sentiment: NEUTRAL 😐 (Confidence: {confidence}%)")

st.markdown("---")
st.caption("Built by [Your Name] | Data source: Skytrax | Models: VADER, TF-IDF + LR, DistilBERT")