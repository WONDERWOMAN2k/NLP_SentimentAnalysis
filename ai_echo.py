# ai_echo.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import spacy
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Setup
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("spaCy model not found. Please install: python -m spacy download en_core_web_sm")
    st.stop()

# Page Config
st.set_page_config(page_title="AI Echo: ChatGPT Sentiment Analyzer", layout="wide")

st.title("🤖 AI Echo: ChatGPT Sentiment Analyzer")
st.write("Upload your ChatGPT review CSV file to analyze sentiments and trends.")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    required_cols = ['review', 'rating', 'date', 'verified_purchase', 'location', 'platform', 'version']
    if not all(col in df.columns for col in required_cols):
        st.error("Missing required columns. Please ensure your CSV contains: " + ", ".join(required_cols))
        st.stop()

    # Preprocess
    stop_words = spacy.lang.en.stop_words.STOP_WORDS

    def preprocess(text):
        doc = nlp(str(text).lower())
        return " ".join([token.text for token in doc if token.is_alpha and token.text not in stop_words])

    df['cleaned_review'] = df['review'].astype(str).apply(preprocess)

    # 1. Sentiment Classification
    def get_sentiment(text):
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            return 'Positive'
        elif polarity < -0.1:
            return 'Negative'
        else:
            return 'Neutral'

    df['sentiment'] = df['cleaned_review'].apply(get_sentiment)

    st.subheader("1. Overall Sentiment Distribution")
    st.bar_chart(df['sentiment'].value_counts())

    # 2. Sentiment vs Rating
    st.subheader("2. Sentiment vs Rating")
    df['polarity'] = df['cleaned_review'].apply(lambda x: TextBlob(x).sentiment.polarity)
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='rating', y='polarity', ax=ax)
    st.pyplot(fig)

    # 3. WordCloud per Sentiment
    st.subheader("3. WordClouds by Sentiment")
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        st.markdown(f"**{sentiment} Reviews**")
        text = " ".join(df[df['sentiment'] == sentiment]['cleaned_review'])
        wordcloud = WordCloud(width=800, height=400).generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    # 4. Sentiment Over Time
    st.subheader("4. Sentiment Trends Over Time")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.to_period('M')
    sentiment_trend = df.groupby(['month', 'sentiment']).size().unstack().fillna(0)
    st.line_chart(sentiment_trend)

    # 5. Sentiment by Verified Purchase
    st.subheader("5. Verified vs Non-Verified Users")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='verified_purchase', hue='sentiment', ax=ax)
    st.pyplot(fig)

    # 6. Review Length vs Sentiment
    st.subheader("6. Review Length vs Sentiment")
    df['review_length'] = df['cleaned_review'].apply(lambda x: len(x.split()))
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='sentiment', y='review_length', ax=ax)
    st.pyplot(fig)

    # 7. Sentiment by Location
    st.subheader("7. Location-based Sentiment")
    loc_sent = df.groupby(['location', 'sentiment']).size().unstack().fillna(0)
    st.bar_chart(loc_sent)

    # 8. Platform-based Sentiment
    st.subheader("8. Platform-based Sentiment")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='platform', hue='sentiment', ax=ax)
    st.pyplot(fig)

    # 9. Version-based Sentiment
    st.subheader("9. App Version-based Sentiment")
    ver_sent = df.groupby(['version', 'sentiment']).size().unstack().fillna(0)
    st.bar_chart(ver_sent)

    # 10. Topic Modeling of Negative Reviews
    st.subheader("10. Topics in Negative Reviews")
    neg_reviews = df[df['sentiment'] == 'Negative']['cleaned_review']
    if len(neg_reviews) > 0:
        cv = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
        dtm = cv.fit_transform(neg_reviews)
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(dtm)
        words = cv.get_feature_names_out()

        for idx, topic in enumerate(lda.components_):
            st.markdown(f"**Topic {idx+1}:**")
            st.write(", ".join([words[i] for i in topic.argsort()[-10:]]))
    else:
        st.info("No negative reviews to analyze for topics.")
