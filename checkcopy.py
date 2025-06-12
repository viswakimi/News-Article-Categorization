import streamlit as st
import torch
import datetime
import pymysql
from sqlalchemy import create_engine
from transformers import BertTokenizer, BertForSequenceClassification

# --- Page Configuration (Must be FIRST) ---
st.set_page_config(page_title="News Article Categorization", layout="wide")

# --- Load model and tokenizer ---
@st.cache_resource
def load_model_and_tokenizer():
    model_path = "D:/App/vscode1/news/bert_tiny_news_classifier_final.pth"
    model = BertForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=4)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# --- Predict Category ---
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax().item()
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    return labels[predicted_class]

# --- Connect to MySQL and Create Table ---
def connect_db():
    engine = create_engine('mysql+pymysql://root:Abcd1234@localhost/news_classifier')
    conn = engine.raw_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_login (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255),
            interaction_time DATETIME,
            input_text TEXT,
            predicted_category VARCHAR(50)
        );
    """)
    conn.commit()
    return conn, cursor

# --- Store user data ---
def store_data(name, time, input_text=None, predicted_category=None):
    try:
        conn, cursor = connect_db()
        cursor.execute("""
            INSERT INTO user_login (name, interaction_time, input_text, predicted_category)
            VALUES (%s, %s, %s, %s)
        """, (name, time, input_text, predicted_category))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Database Error: {e}")


# Title
st.title("📰 News Article Categorization using BERT")

# Initialize session state
if "name_entered" not in st.session_state:
    st.session_state.name_entered = False
if "name" not in st.session_state:
    st.session_state.name = ""

# Name input section
if not st.session_state.name_entered:
    name = st.text_input("Enter your name:")
    if st.button("Start Classification", key="start"):
        if name.strip() == "":
            st.warning("Please enter your name.")
        else:
            st.session_state.name = name
            st.session_state.name_entered = True
            st.rerun()  # Immediately rerun to show next section
else:
    st.success(f"✅ Welcome, {st.session_state.name}! You can now classify news articles.")

    # Show classification input section
    user_input = st.text_area("✍️ Enter news article text here")

    # Predict button
    if st.button("🧠 Categorize"):
        if user_input.strip() == "":
            st.warning("Please enter a news article.")
        else:
            category = predict(user_input)
            st.success(f"✅ Predicted Category: **{category}**")
            store_data(st.session_state.name, datetime.datetime.now(), user_input, category)

    # Back button to change name
    if st.button("🔙 Back"):
        st.session_state.name_entered = False
        st.session_state.name = ""
        st.rerun()
