import streamlit as st
from openai import OpenAI
import spacy

# Load OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load Spacy model
nlp = spacy.load("en_core_web_sm")

# Predefined stock market-related entities
stock_entities = ["Apple", "Google", "Microsoft", "Tesla", "NASDAQ", "Dow Jones", "S&P 500", "Bitcoin", "Ethereum"]

# Function to perform sentiment analysis using OpenAI GPT
def analyze_sentiment(review, category):
    prompt = f"""Analyze the sentiment of the following {category} statement in the context of stock market performance. \
Classify it as Positive (bullish), Negative (bearish), or Neutral:\n\nStatement: {review}"""

    response = client.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial sentiment analysis expert. Respond only with the classification."},
            {"role": "user", "content": prompt},
        ]
    )

    sentiment = response['choices'][0]['message']['content']
    return sentiment.strip()

# Function to extract entities using spaCy
def extract_entities(review):
    doc = nlp(review)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    matched_stocks = [ent for ent in entities if ent[0] in stock_entities]
    return matched_stocks if matched_stocks else entities

# Streamlit UI
def main():
    st.set_page_config(page_title="Stock Sentiment Chatbot", layout="centered")
    st.title("📊 Stock Market Sentiment Analyzer")
    st.write("Analyze the sentiment and extract financial entities from your stock market-related statements.")

    category = st.selectbox("Select Category", ["Stock", "Index", "Crypto", "Economy", "Other"])
    review = st.text_area(f"Enter your market statement about {category.lower()}:")

    if st.button("Analyze"):
        if review.strip():
            st.markdown("### 🧠 Sentiment Analysis")
            sentiment = analyze_sentiment(review, category)
            st.success(sentiment)

            st.markdown("### 🏷️ Named Entity Recognition")
            entities = extract_entities(review)
            st.write(entities)
        else:
            st.warning("Please enter a valid statement.")

if __name__ == "__main__":
    main()
