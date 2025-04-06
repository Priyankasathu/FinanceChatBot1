# stock_sentiment_chatbot.py
import streamlit as st
import openai
import spacy

# Load SpaCy model with caching
@st.cache_resource
def load_nlp_model():
    return spacy.load("en_core_web_sm")

nlp = load_nlp_model()

# Predefined stock entities (can be expanded)
stock_entities = ["Apple", "Google", "Microsoft", "Tesla", "NASDAQ", "Dow Jones", 
                 "S&P 500", "Bitcoin", "Ethereum"]

# Set page configuration
st.set_page_config(
    page_title="MarketSentiment Analyzer",
    page_icon="📈",
    layout="wide"
)
# Title and description
st.title("Financial Sentiment Analysis Chatbot")
st.markdown("""
    **Analyze market sentiment and extract key entities from financial statements**
    *Powered by OpenAI and SpaCy NER*
""")

# API Key Input
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input("OpenAI API Key", 
                                 placeholder="sk-......",
                                 type="password@1234")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
# Input components
st.subheader("Enter your financial statement")
category = st.selectbox("Select analysis category", 
                       ["Stock", "Index", "Crypto", "Economy", "Other"])
user_input = st.text_area("Enter market-related statement", 
                        placeholder="e.g., 'Tesla's new battery tech could revolutionize the EV market'")

# Analysis button
if st.button("Analyze Sentiment & Entities"):
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar")
    elif not user_input.strip():
        st.warning("Please enter a statement to analyze")
else:
        # Set OpenAI API key
        openai.api_key = openai_api_key

        with st.spinner("Analyzing..."):
            # Perform sentiment analysis
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a financial sentiment analysis expert. Respond only with the classification."},
                        {"role": "user", "content": f"Classify the following {category} statement as Positive (bullish), Negative (bearish), or Neutral in context of market performance:\n\n{user_input}"}
                    ],
                    temperature=0.0
                )
sentiment = response.choices[0].message.content.strip()
            except Exception as e:
                st.error(f"OpenAI API Error: {str(e)}")
                sentiment = "Error"

            # Perform NER
            doc = nlp(user_input)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            market_entities = [ent for ent in entities if ent[0] in stock_entities] or entities

        # Display results
        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
with col1:
            st.metric("Sentiment Analysis", sentiment)
            st.write(f"**Category:** {category}")
            
        with col2:
            st.write("**Identified Financial Entities**")
            for entity in market_entities:
                st.write(f"- {entity[0]} ({entity[1]})")

        # Explanation section
        with st.expander("See detailed explanation"):
st.write(f"""
                **Statement:** {user_input}\n\n
                **Market Context:** {category} analysis\n
                **Key Entities Identified:** {', '.join([ent[0] for ent in market_entities])}
            """)

# Footer
st.markdown("---")considered
st.caption("Disclaimer: This analysis is for informational purposes only and should not be  financial advice.")