# ai_echo.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from wordcloud import WordCloud
from textblob import TextBlob
import spacy
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# --- Download required resources ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("🤖 AI Echo: ChatGPT User Review Sentiment Analyzer")
st.markdown("Analyze ChatGPT reviews using NLP, sentiment analysis, and topic modeling.")

# --- File Upload ---
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- Preprocess Text ---
    @st.cache_data
    def preprocess(text):
        doc = nlp(str(text).lower())
        stop_words = spacy.lang.en.stop_words.STOP_WORDS
        return " ".join([token.text for token in doc if token.is_alpha and token.text not in stop_words])

    df['cleaned_review'] = df['review'].astype(str).apply(preprocess)

    # --- Sentiment Classification ---
    def get_sentiment(text):
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            return 'Positive'
        elif polarity < -0.1:
            return 'Negative'
        else:
            return 'Neutral'

    df['sentiment'] = df['cleaned_review'].apply(get_sentiment)

    # --- Sentiment Distribution ---
    st.subheader("1️⃣ Sentiment Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='sentiment', ax=ax)
    ax.set_title('Overall Sentiment Distribution')
    st.pyplot(fig)

    # --- Sentiment vs Rating ---
    st.subheader("2️⃣ Sentiment Polarity vs Rating")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='rating', y=df['cleaned_review'].apply(lambda x: TextBlob(x).sentiment.polarity), ax=ax)
    ax.set_title("Rating vs Sentiment Polarity")
    st.pyplot(fig)

    # --- WordClouds ---
    st.subheader("3️⃣ WordClouds by Sentiment")
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        text = " ".join(df[df['sentiment'] == sentiment]['cleaned_review'])
        wordcloud = WordCloud(width=800, height=400).generate(text)
        st.markdown(f"#### {sentiment} Reviews WordCloud")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    # --- Sentiment Over Time ---
    st.subheader("4️⃣ Sentiment Trends Over Time")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.to_period('M')
    sentiment_trend = df.groupby(['month', 'sentiment']).size().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(12, 6))
    sentiment_trend.plot(marker='o', ax=ax)
    ax.set_title('Sentiment Over Time')
    ax.set_ylabel('Number of Reviews')
    st.pyplot(fig)

    # --- Verified Purchase ---
    st.subheader("5️⃣ Sentiment by Verified Purchase")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='verified_purchase', hue='sentiment', ax=ax)
    ax.set_title('Sentiment by Verified Purchase')
    st.pyplot(fig)

    # --- Review Length vs Sentiment ---
    st.subheader("6️⃣ Review Length vs Sentiment")
    df['review_length'] = df['cleaned_review'].apply(lambda x: len(x.split()))
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='sentiment', y='review_length', ax=ax)
    ax.set_title('Review Length vs Sentiment')
    st.pyplot(fig)

    # --- Location-Based Sentiment ---
    st.subheader("7️⃣ Location-Based Sentiment")
    location_sentiment = df.groupby(['location', 'sentiment']).size().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(12, 6))
    location_sentiment.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Sentiment by Location')
    ax.set_ylabel('Number of Reviews')
    st.pyplot(fig)

    # --- Platform-Based Sentiment ---
    st.subheader("8️⃣ Platform-Based Sentiment")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='platform', hue='sentiment', ax=ax)
    ax.set_title('Sentiment by Platform')
    st.pyplot(fig)

    # --- Version-Based Sentiment ---
    st.subheader("9️⃣ Version-Based Sentiment")
    version_sentiment = df.groupby(['version', 'sentiment']).size().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(12, 6))
    version_sentiment.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Sentiment by App Version')
    ax.set_ylabel('Number of Reviews')
    st.pyplot(fig)

    # --- Topic Modeling for Negative Reviews ---
    st.subheader("🔟 Topic Modeling on Negative Reviews")
    negative_reviews = df[df['sentiment'] == 'Negative']['cleaned_review']
    cv = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    dtm = cv.fit_transform(negative_reviews)
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(dtm)

    words = cv.get_feature_names_out()
    for index, topic in enumerate(lda.components_):
        st.markdown(f"**Topic #{index + 1}:**")
        topic_words = [words[i] for i in topic.argsort()[-10:]]
        st.write(", ".join(topic_words))

else:
    st.warning("👆 Please upload a CSV file with ChatGPT reviews to begin.")
