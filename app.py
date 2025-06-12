import streamlit as st
import torch
import datetime
import pymysql
import os
import boto3
from sqlalchemy import create_engine
from transformers import BertTokenizer, BertForSequenceClassification

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="News Article Categorization", layout="wide")

# --- Load model and tokenizer from S3 ---
@st.cache_resource
def load_model_and_tokenizer():
    s3 = boto3.client('s3')

    bucket_name = "newsapp-model-storage"
    model_key = "bert_tiny_news_classifier_final.pth"
    local_model_path = "/tmp/bert_tiny_news_classifier_final.pth"

    # Download the model from S3 only once
    if not os.path.exists(local_model_path):
        s3.download_file(bucket_name, model_key, local_model_path)

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
    model = BertForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=4)
    model.load_state_dict(torch.load(local_model_path, map_location=torch.device("cpu")))
    model.eval()
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# --- Predict Category Function ---
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax().item()
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    return labels[predicted_class]

# --- Connect to RDS MySQL and Create Table ---
def connect_db():
    # Replace with your actual RDS credentials and endpoint
    db_user = "admin"
    db_password = "Kimira1308"
    db_endpoint = "newsappdb.cctak66kwb3e.us-east-1.rds.amazonaws.com"
    db_name = "newsappdb"

    engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_endpoint}:3306/{db_name}")
    conn = engine.raw_connection()
    cursor = conn.cursor()

    # Create table if not exists
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

# --- Store User Interaction to RDS ---
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
        st.error(f"❌ Database Error: {e}")

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