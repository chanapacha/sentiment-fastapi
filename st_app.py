import streamlit as st
import pandas as pd
from SentimentAnalyzer import *
from utils import split_fn

# Initialize SentimentAnalyzer
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = SentimentAnalyzer()

# Function to analyze sentiment of text
@st.cache_data
def analyze_sentiment_text(text):
    sentiment_pred, sentiment_conf, trigger = st.session_state.analyzer.sentiment_classify(text)
    return sentiment_pred, sentiment_conf, trigger

# Function to analyze sentiment of uploaded Excel file
@st.cache_data
def analyze_sentiment_excel(file):
    df = pd.read_excel(file)
    results = []
    for text in df['text']:
        sentiment_pred, sentiment_conf, trigger = st.session_state.analyzer.sentiment_classify(text)
        results.append({"sentiment": sentiment_pred, "confidence": sentiment_conf, "trigger": trigger})
    df_result = pd.concat([df, pd.DataFrame(results)], axis=1)
    return df_result

# Main Streamlit app
def main():
    st.title('Sentiment Analysis Web App')
    st.write('Welcome to the Sentiment Analysis Web App!')
    st.write('Created by Chanapa Chareesan')

    # Text Input Section
    st.subheader('Text Input')
    text_input = st.text_input('Enter your text:')
    if st.button('Analyze Text'):
        sentiment_pred, sentiment_conf, trigger = analyze_sentiment_text(text_input)
        st.write('Sentiment:', sentiment_pred)
        st.write('Confidence:', sentiment_conf)
        st.write('Trigger:', trigger)

    # File Upload Section
    st.subheader('File Upload')
    st.write('Please ensure that the uploaded Excel file contains a column named :blue[text]')
    st.write('For example:')
    st.write(pd.DataFrame({
    'text': ['แอปตัวนี้ทำไมมันใช้งานยากจัง', 'ที่นี่ดูแลดีใส่ใจลูกค้าดีมากเลยครั้งหน้าจะมาใช้บริการอีกแน่นอน', 'ต้องการร้องเรียนพฤติกรรมของพนักงาน']
}))
    uploaded_file = st.file_uploader('Upload Excel file', type=['xlsx', 'xls'])
    if uploaded_file is not None:
        if st.button('Analyze Excel'):
            df_result = analyze_sentiment_excel(uploaded_file)
            st.write('Analysis Results:')
            st.write(df_result)

if __name__ == '__main__':
    main()

