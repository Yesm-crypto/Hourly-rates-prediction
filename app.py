from flask import Flask, render_template, request, send_file
import jsonify
import requests
import pickle
import numpy as np
import pandas as pd
import sklearn
import re
from string import punctuation

app = Flask(__name__)
model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))
w2v_model = pickle.load(open('word2vec_model.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['GET'])
def predict():
    return render_template('predict.html')

@app.route("/upload", methods=['GET'])
def upload():
    sample_file = pd.ExcelFile("sample.xlsx")
    df = pd.read_excel(sample_file,"sample")
    html = df.to_html()
    return render_template('upload.html',sample_file=html)

@app.route("/submit", methods=['POST'])
def submit():
    try:
        if request.method=="POST":
            global filename
            file = request.files["file"]
            margin = int(request.form["Margin"])
            # print(margin)
            # print(uploaded_file.filename)
            try:
                if file.filename.split(".")[-1]=="csv":
                    # print("This is a csv file")
                    df = pd.read_csv(file)
                    raw_df = df.copy()
                elif file.filename.split(".")[-1]=="xlsx":
                    # print("This is an excel file")
                    df = pd.read_excel(file.read())
                    raw_df = df.copy()
            except Exception as e:
                sample_file = pd.ExcelFile("sample.xlsx")
                df = pd.read_excel(sample_file,"sample")
                html = df.to_html()
                errors = f"""<h1 class="err"> Something is wrong with the file! <br> Please confirm whether the above criterias are fulfilled or not! </h1> <h4> Error: {e} </h4>"""
                return render_template("upload.html",errors=errors,sample_file=html)
            
            # Checking the columns similarity
            columns = raw_df.columns
            ideal_cols = ["class","role","campaign type","average salary per annum","city","cost of living"]
            similar_cols = True
            for index, item in enumerate(ideal_cols):
                if ideal_cols[index]!=columns[index]:
                    similar_cols = False
            if not similar_cols:
                sample_file = pd.ExcelFile("sample.xlsx")
                df = pd.read_excel(sample_file,"sample")
                html = df.to_html()
                errors = """<h1 class="err">Column headings should be exactly same!<br>Take this sample table as reference.</h1>"""
                return render_template("upload.html",errors=errors,sample_file=html)
            
            #Dropping null values
            raw_df = raw_df.dropna()
            try:
                # applying function to convert the rates into integers
                raw_df["average salary per annum"] = raw_df["average salary per annum"].apply(conv_to_int)
                # applying function to calculate hourly rates
                raw_df["rph_feature"] = raw_df["average salary per annum"].apply(rate_calc)
                # preprocessing the text data
                raw_df["class"] = raw_df["class"].apply(preprocess)
                raw_df["role"] = raw_df["role"].apply(preprocess)
                raw_df["campaign type"] = raw_df["campaign type"].apply(preprocess)
                raw_df["city"] = raw_df["city"].apply(preprocess)

                types = raw_df["campaign type"].unique()
                if len(types)>3:
                    sample_file = pd.ExcelFile("sample.xlsx")
                    df = pd.read_excel(sample_file,"sample")
                    html = df.to_html()
                    errors = f"""<h3>The model is trained on 3 campaign types(DTC,HCP and IC). <br>But in the uploaded file, there are {len(types)} campaign types. <br>  {', '.join(types).upper()} </h3>"""
                    return render_template("upload.html",errors=errors,sample_file=html)

                def dummy_hcp(types):
                    hcp=["health care professional","health care professionals","hcp"]
                    dtc=["direct to consumer","direct to consumers","dtc"]
                    ic=["integrated communication","integrated communications","ic"]
                    if types in hcp:
                        return 1
                    elif types in dtc:
                        return 0
                    elif types in ic:
                        return 0
            
                def dummy_ic(types):
                    hcp=["health care professional","health care professionals","hcp"]
                    dtc=["direct to consumer","direct to consumers","dtc"]
                    ic=["integrated communication","integrated communications","ic"]
                    if types in hcp:
                        return 0
                    elif types in dtc:
                        return 0
                    elif types in ic:
                        return 1
                    
                def dummy_ny(city):
                    ny=["new york","New York","ny"]
                    la=["los angeles","la","los angles"]
                    ch=["chicago","Chicago","chi"]
                    if city in ny:
                        return 1
                    elif city in la:
                        return 0
                    elif city in ch:
                        return 0
            
                def dummy_la(city):
                    ny=["new york","New York","ny"]
                    la=["los angeles","la","los angles"]
                    ch=["chicago","Chicago","chi"]
                    if city in ny:
                        return 0
                    elif city in la:
                        return 1
                    elif city in ch:
                        return 0
                # dummy variables for campaign types
                raw_df["health care professional"] = raw_df["campaign type"].apply(dummy_hcp)
                raw_df["integrated communications"] = raw_df["campaign type"].apply(dummy_ic)
                # dummy variables for cities
                raw_df["new york"] = raw_df["city"].apply(dummy_ny)
                raw_df["los angeles"] = raw_df["city"].apply(dummy_la)

                # converting class and roles into vectors
                raw_df["class"] = raw_df["class"].apply(modulus_of_wv)
                raw_df["role"] = raw_df["role"].apply(modulus_of_wv)
            except Exception as e:
                sample_file = pd.ExcelFile("sample.xlsx")
                df = pd.read_excel(sample_file,"sample")
                html = df.to_html()
                errors = f"""<h1 class="err"> Something is wrong with the data types of elements! <br> Please take this sample table as reference! </h1> <h4> Error: {e} </h4>"""
                return render_template("upload.html",errors=errors,sample_file=html)

            # Making final df for prediction
            final_df = raw_df[["class","role","rph_feature",'health care professional', 'integrated communications',"new york",'los angeles','cost of living']]
            # Let's have predictions
            predictions = model.predict(final_df)
            # Make a df out of predictions
            pred_df = pd.DataFrame({"Predicted Rates":predictions})
            # Function to calculate margin
            def with_margin(x):
                wm = x + int((margin/100)*x)
                return wm
            # Applying the function to calculate margin and creating a new column our of it
            pred_df["Rates with Margin"] = pred_df["Predicted Rates"].apply(with_margin)
            # Concatenating the dataframes
            show_df = pd.concat((df,pred_df),axis=1)
            # Returning the dataframe to the user
            if file.filename.split(".")[-1]=="csv":
                filename="rhpreds.csv"
                show_df.to_csv(filename,index=False)
            elif file.filename.split(".")[-1]=="xlsx":
                filename="rhpreds.xlsx"
                show_df.to_excel(filename,index=False)
            html=show_df.head(10).to_html()
            # df = pd.ExcelFile()
            return render_template('download.html',sample_file=html)
        else:
            return render_template("index.html")
    except Exception as e:
            sample_file = pd.ExcelFile("sample.xlsx")
            df = pd.read_excel(sample_file,"sample")
            html = df.to_html()
            errors = f"""<h1 class="err"> Something is wrong with the file! <br> Please take this sample table as reference! </h1> <h4> Error: {e} </h4>"""
            return render_template("upload.html",errors=errors,sample_file=html)

@app.route("/predictions", methods=['POST'])
def predictions():
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
            return render_template('predict.html',salary_error_text="Sorry! Enter the labels in integer format!")

# "Class","Role","rph_feature",'health care professional', 'integrated communications',"New York",'Los Angles','Cost of living'        
        prediction=model.predict([[Class,Role,rph_feature,c_hcp,c_ic,City_Ny,City_La,col]])
        output_1=int(prediction[0])
        margin=int(request.form['Margin'])
        output = output_1 + int((margin/100)*output_1)
        if output<0:
            return render_template('predict.html')
        else:
            return render_template('predict.html',Class=c.capitalize(),Role = r.capitalize(),City=City,average_salary=average_salary,COL=col,output_1 = output_1,output=output,Type=Campaign)
    else:
        return render_template('predict.html')

@app.route("/download-file/")
def download():
    global filename
    return send_file(filename, attachment_filename=filename, as_attachment=True)

# Preprocessing
def preprocess(content):
    content = content.replace(","," ").lower().replace("$","")
    text = re.sub(r'\[[0-9]*\]',' ',content)
    text = re.sub(r'\s+',' ',text)
    text = re.sub(r'\d',' ',text)
    text = re.sub(r'\s+',' ',text)
    text = "".join([letter for letter in text if letter not in punctuation])
    text = text.rstrip().lower()
    return text

# convert the rates to integers
def conv_to_int(x):
    try:
        t_n = x.rstrip().replace(",","").replace("$","")
    except:
        t_n = x
    n = int(t_n)
    return n
# Calculation of hourly rates from yearly rates
def rate_calc(x):
    rate = int(int(x)/2080)
    return rate

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


