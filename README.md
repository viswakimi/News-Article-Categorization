# News-Article-Categorization

# News Article Categorization Application using AWS Services

## 📌 Project Overview
This project deploys a fine-tuned **text classification model** to categorize news articles into predefined categories such as **Sports, Politics, Technology, and Entertainment**. The application is built using **Hugging Face Transformers, Streamlit, and AWS Services (S3, EC2, RDS)** for hosting and storage.

## 🚀 Tech Stack
- **Machine Learning**: Hugging Face (BERT, RoBERTa)
- **Deployment**: AWS S3, EC2, RDS
- **Web Application**: Streamlit
- **Database**:  Amazon RDS
- **Programming Language**: Python

## 🎯 Project Workflow
1. **Fine-tune the model** using the AG News Dataset
2. **Upload the model** to an AWS S3 bucket
3. **Host the application** on an EC2 instance
4. **Log user interactions** in an Amazon RDS database
5. **Deploy the web application** using Streamlit

## 📂 Dataset
- AG News Classification Dataset from Kaggle
- Link: [AG News Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

## 🛠️ Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/viswakimi/News-Article-Categorization
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train & Save the Model
```bash
python train.py
```

### 4️⃣ Upload Model to S3
```bash
python upload_to_s3.py
```

### 5️⃣ Deploy the Web App on EC2
- Launch an **EC2 instance** and install dependencies:
```bash
pip install transformers torch boto3 streamlit
```
- Run the Streamlit app:
```bash
streamlit run app.py --server.port 8501 --server.enableCORS false
```

## 📊 API & Web Application
- The web application allows users to input news articles for classification.
- The model predicts the category and logs user inputs into an RDS database.

## 🔐 Security Measures
- **IAM Roles**: Secure access to AWS resources
- **Security Groups**: Restrict access to the EC2 instance
- **Database Encryption**: Secure RDS data storage

## 📌 Future Enhancements
- Integrate **Docker** for containerization
- Implement **FastAPI** for a REST API
- Add **real-time analytics** on user interactions


