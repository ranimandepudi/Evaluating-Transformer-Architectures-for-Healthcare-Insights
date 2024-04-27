from flask import Flask, render_template,request,redirect,session
import os
import joblib
import pandas as pd
import torch
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

app = Flask(__name__)
app.secret_key=os.urandom(24)
# Model saved with Keras model.save()
MODEL_PATH = './distilbert-drug-review-model'

TOKENIZER_PATH ='./distilbert-drug-review-tokenizer'

DATA_PATH ='drugsComTrain.csv'
LABEL_ENCODER_PATH = 'label_encoder.pkl'


# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)  # Ensure label_encoder is saved after training

lemmatizer = WordNetLemmatizer()
stop = stopwords.words('english')


@app.route('/')
def login():
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

@app.route('/login_validation', methods=['POST'])
def login_validation():
    username = request.form.get('username')
    password = request.form.get('password')
    
    if username == "rani@gmail.com" and password == "nlp":
        session['user_id'] = username
        return redirect('/index')
    else:
        err = "Kindly Enter valid User ID/ Password"
        return render_template('login.html', lbl=err)

@app.route('/index')
def index():
    if 'user_id' in session:
        return render_template('home.html')
    else:
        return redirect('/')

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        raw_text = request.form['rawtext']
        
        if raw_text:
            clean_text = cleanText(raw_text)
            encoded_text = tokenizer(clean_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
            with torch.no_grad():
                outputs = model(**encoded_text)
            predictions = torch.argmax(outputs.logits, dim=-1)
            predicted_condition = label_encoder.inverse_transform(predictions.numpy())[0]
            
            df = pd.read_csv(DATA_PATH)
            top_drugs = top_drugs_extractor(predicted_condition, df)
            
            return render_template('predict.html', rawtext=raw_text, result=predicted_condition, top_drugs=top_drugs)
        else:
            return render_template('predict.html', error="No text provided for prediction.")

def cleanText(raw_review):
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if not w in stop]
    lemmitized_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    return ' '.join(lemmitized_words)

def top_drugs_extractor(condition, df):
    df_top = df[(df['rating'] >= 9) & (df['usefulCount'] >= 100)].sort_values(by=['rating', 'usefulCount'], ascending=False)
    drug_list = df_top[df_top['condition'] == condition]['drugName'].head(3).tolist()
    return drug_list

if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=8080)