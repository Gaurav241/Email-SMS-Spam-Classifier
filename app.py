import os
import json
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from streamlit_lottie import st_lottie
import time

# ========== CRITICAL NLTK SETUP ==========
@st.cache_resource
def setup_nltk():
    try:
        # Set custom path for NLTK data in Streamlit Cloud
        nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
        os.makedirs(nltk_data_path, exist_ok=True)
        nltk.data.path.append(nltk_data_path)
        
        # Download required data if not found
        if not nltk.data.find('tokenizers/punkt'):
            nltk.download('punkt', download_dir=nltk_data_path)
        if not nltk.data.find('corpora/stopwords'):
            nltk.download('stopwords', download_dir=nltk_data_path)
        
        # Verify punkt is accessible
        nltk.data.find('tokenizers/punkt')
        return True
    except Exception as e:
        st.error(f"NLTK setup failed: {str(e)}")
        return False

if not setup_nltk():
    st.stop()

# ========== MODEL LOADING ==========
@st.cache_resource
def load_models():
    try:
        with open('vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return tfidf, model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

tfidf, model = load_models()

# ========== TEXT PROCESSING ==========
ps = PorterStemmer()

def transform_text(text: str) -> str:
    try:
        text = text.lower()
        text = nltk.word_tokenize(text)
        text = [word for word in text if word.isalnum()]
        text = [word for word in text if word not in stopwords.words('english')]
        text = [ps.stem(word) for word in text]
        return ' '.join(text)
    except Exception as e:
        st.error(f"Text processing error: {str(e)}")
        return ""

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="Spam Classifier", layout="centered")

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📧 Spam Message Classifier")
    st.markdown("Identify spam messages with **97.2% accuracy**")

with col2:
    try:
        lottie_spam = json.load(open("side_image.json"))
        st_lottie(lottie_spam, height=150)
    except:
        st.warning("Animation not available")

# Input
input_msg = st.text_area("Enter your message:", height=150, 
                        placeholder="Paste an email or SMS message here...")

if st.button("Analyze", type="primary"):
    if not input_msg.strip():
        st.warning("Please enter a message")
    else:
        with st.spinner("Analyzing..."):
            try:
                processed = transform_text(input_msg)
                vector = tfidf.transform([processed])
                result = model.predict(vector)[0]
                
                if result == 1:
                    st.error("🚨 This is SPAM")
                else:
                    st.success("✅ This is NOT spam")
                    
                st.info(f"Processed text: _{processed}_")
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

st.caption("Note: This model has 97.2% accuracy on test data")
