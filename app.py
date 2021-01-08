from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import re
from string import punctuation

app = Flask(__name__)
model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))
w2v_model = pickle.load(open('word2vec_model.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    Fuel_Type_Diesel=0
    if request.method == 'POST':
        try:
            c = preprocess(request.form['Class'])
            # print(c)
            Class = modulus_of_wv(c)
            r = preprocess(request.form['Role'])
            Role= modulus_of_wv(r)
        except:
            return render_template('index.html',cr_error_text="Sorry! Enter the labels in text format!")
        Campaign=request.form['Campaign']
        # print(Campaign)
        if Campaign=='HCP':
            c_hcp = 1 
            c_ic = 0
        elif Campaign=='IC':
            c_hcp = 0 
            c_ic = 1
        else:
            c_hcp = 0 
            c_ic = 0
        
        City=request.form['City']
        if City=='New York':
            City_Ny = 1
            City_La = 0
        elif City=="Los Angeles":
            City_Ny = 0
            City_La = 1
        else:
            City_Ny = 0
            City_La = 0
        try:
            average_salary = int(request.form['Average salary'])
            rph_feature = int(average_salary/2080)
            col = float(request.form['col'])
            # print(average_salary,rph_feature,col)
        except:
            return render_template('index.html',salary_error_text="Sorry! Enter the labels in integer format!")

# "Class","Role","rph_feature",'health care professional', 'integrated communications',"New York",'Los Angles','Cost of living'        
        prediction=model.predict([[Class,Role,rph_feature,c_hcp,c_ic,City_Ny,City_La,col]])
        output_1=int(prediction[0])
        margin=int(request.form['Margin'])
        output = output_1 + int((margin/100)*output_1)
        if output<0:
            return render_template('index.html')
        else:
            return render_template('index.html',Class=c.capitalize(),Role = r.capitalize(),City=City,average_salary=average_salary,COL=col,output_1 = output_1,output=output,Type=Campaign)
    else:
        return render_template('index.html')

# Preprocessing
def preprocess(content):
    content = content.replace(","," ").lower().replace("$","")
    text = re.sub(r'\[[0-9]*\]',' ',content)
    text = re.sub(r'\s+',' ',text)
    text = re.sub(r'\d',' ',text)
    text = re.sub(r'\s+',' ',text)
    text = "".join([letter for letter in text if letter not in punctuation])
    text = text.rstrip()
    return text

# Finding the average feature vector representing the whole sentence
num_features = 100
index2word_set = set(w2v_model.wv.index2word)
def avg_feature_vector(sentence, model, num_features, index2word_set):
    words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model.wv[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec
# Finding the modulus of word vectors
def modulus_of_wv(words):
    global w2v_model
    global index2word_set
    global num_features
    wv = avg_feature_vector(words,w2v_model,num_features,index2word_set)
    mod = np.linalg.norm(wv)
    return mod

if __name__=="__main__":
    app.run(debug=True)


