import streamlit as st
import torch
from transformers import BertTokenizer
from model import BertLSTM

# Load model and tokenizer
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertLSTM()
    model.load_state_dict(torch.load('models/bert_lstm_model.pt', map_location=device))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('models/tokenizer')
    return model, tokenizer, device

# Sentiment prediction
def predict_sentiment(text, model, tokenizer, device):
    encoding = tokenizer(
        text,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        inputs = {
            'input_ids': encoding['input_ids'].to(device),
            'attention_mask': encoding['attention_mask'].to(device)
        }
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
    return probabilities

# Streamlit app
def main():
    st.set_page_config(page_title="Movie Sentiment Analyzer", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
        .stApp { background-color: #f0f2f6; }
        .title { color: #1f77b4; text-align: center; }
        .positive { color: green; font-weight: bold; }
        .negative { color: red; font-weight: bold; }
        .neutral { color: orange; font-weight: bold; }
        .review-box { background-color: white; border-radius: 10px; padding: 20px; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1 class='title'>🎬 Movie Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("### Powered by BERT-LSTM Hybrid Model")
    
    # Load model
    model, tokenizer, device = load_model()
    
    # Input section
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Analyze Your Movie Review")
        review = st.text_area("Enter your movie review:", height=200, 
                             placeholder="This movie was absolutely fantastic...")
        
        if st.button("Analyze Sentiment", use_container_width=True):
            if review.strip() == "":
                st.warning("Please enter a movie review")
            else:
                with st.spinner("Analyzing..."):
                    probs = predict_sentiment(review, model, tokenizer, device)
                    positive_prob = probs[1]
                    negative_prob = probs[0]
                    
                    # Display results
                    if positive_prob > 0.7:
                        sentiment = "POSITIVE 😊"
                        color_class = "positive"
                    elif negative_prob > 0.7:
                        sentiment = "NEGATIVE 😞"
                        color_class = "negative"
                    else:
                        sentiment = "NEUTRAL 😐"
                        color_class = "neutral"
                    
                    st.markdown(f"<div class='review-box'>", unsafe_allow_html=True)
                    st.subheader("Analysis Results")
                    
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric("Sentiment", f"<span class='{color_class}'>{sentiment}</span>", 
                                 unsafe_allow_html=True)
                        
                    with col_res2:
                        st.metric("Confidence", f"{max(positive_prob, negative_prob)*100:.1f}%")
                    
                    # Confidence bars
                    st.progress(positive_prob, text=f"Positive: {positive_prob*100:.1f}%")
                    st.progress(negative_prob, text=f"Negative: {negative_prob*100:.1f}%")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.subheader("Sample Movie Reviews")
        st.markdown("""
        - **Positive**: "This film exceeded all expectations! The cinematography was breathtaking and the performances were Oscar-worthy."
        - **Negative**: "A complete waste of time. Poorly written script with wooden acting throughout. I want my money back."
        - **Neutral**: "The visual effects were impressive, but the plot felt derivative and predictable. Worth a rental."
        """)
        
        st.divider()
        st.subheader("How It Works")
        st.markdown("""
        1. Hybrid BERT-LSTM model extracts deep semantic features
        2. Trained on 50,000 IMDB movie reviews
        3. Classifies sentiment with 94.7% accuracy
        4. Shows prediction confidence levels
        """)

if __name__ == "__main__":
    main()