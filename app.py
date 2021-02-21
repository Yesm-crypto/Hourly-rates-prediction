from flask import Flask, render_template, Response, request, send_file, redirect,url_for
import flask_progress_bar.flask_progress_bar as FPB
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename
import jsonify
import requests
import pickle
import numpy as np
import pandas as pd
import sklearn
from scipy import spatial
import re
import json
from string import punctuation
import threading

app = Flask(__name__)

# Configuring the mail server
app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'hourlyratesprediction@gmail.com'
app.config['MAIL_PASSWORD'] = 'thisishourlyratesprediction'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
app_token="ijklmnop7654890"
mail = Mail(app)

# Loading different models
model = pickle.load(open('./models/random_forest_regression_model.pkl', 'rb'))
w2v_model = pickle.load(open('./models/word2vec_model.pkl', 'rb'))
citycheck_model = pickle.load(open('./models/citycheck_model.pkl', 'rb'))
rolecheck_model = pickle.load(open('./models/rolescheck_model.pkl', 'rb'))

# Setting number features of model and defining index2word sets of each w2v model
num_features = 100
index2word_set_w2v = set(w2v_model.wv.index2word)
index2word_set_cities = set(citycheck_model.wv.index2word)
index2word_set_roles = set(rolecheck_model.wv.index2word)

# some global variables
errors = False
html = None
raw_df = None
filename = ""
margin = 0

# Different routes
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



def process_file(email):
    global errors
    global html
    global filename
    global raw_df
    global  margin
    file_path = "./uploads/" + filename
    # print(file_path)
    # Checking the extension of the file
    try:
        if filename.split(".")[-1]=="csv":
            # print("This is a csv file")
            df = pd.read_csv(file_path)
            raw_df = df.copy()
        elif filename.split(".")[-1]=="xlsx":
            # print("This is an excel file")
            df = pd.read_excel(file_path)
            raw_df = df.copy()
        # print(raw_df.columns)
        # yield [5,100]
    except Exception as e:
        errors = f"""<h1 class="err"> Something is wrong with the file! <br> Please confirm whether the above criterias are fulfilled or not! </h1> <h4> Error: {e} </h4>"""
        # yield [100,100]
        return

    # Checking the columns similarity
    columns = raw_df.columns
    # print(columns)
    ideal_cols = ["class","role","campaign type","city"]
    similar_cols = True
    for index, item in enumerate(ideal_cols):
        if ideal_cols[index]!=columns[index]:
            similar_cols = False
    if not similar_cols:
        errors = """<h1 class="err">Column headings should be exactly same!<br>Take this sample table as reference.</h1>"""
        # yield [100,100]
        return
    print("Column Similarity Checked!")
    

    # yield [10,100]
    # preprocessing process begins
    try:
        #Dropping null values
        raw_df = raw_df.dropna()   
        try:
            # preprocessing the text data
            raw_df["class"] = raw_df["class"].apply(preprocess)
            raw_df["role"] = raw_df["role"].apply(preprocess)
            raw_df["campaign type"] = raw_df["campaign type"].apply(preprocess)
            raw_df["city"] = raw_df["city"].apply(preprocess)
            # yield [15,100]
            # figuring our average salaries of given cities
            # loading json file
            input_file_col = open("./files/cit_col.json")
            input_file_avs = open("./files/cit_salary.json")
            city_col_data = json.load(input_file_col)
            avs_data = json.load(input_file_avs)

            raw_roles = raw_df["role"]
            raw_cities = raw_df["city"]
            # yield [25,100]
            # data for new columns 
            # average salary column
            avs_col = []
            # cost of living column
            col_col = []
            # yield [55,100]
            for role, city in zip(raw_roles, raw_cities):
                sim_dict_city = average_similarity_dict(city,mode="c")
                sim_dict_role = average_similarity_dict(role,mode="r")
                
                for key in sim_dict_city.keys():
                    sim_city = key
                    break
                for key in sim_dict_role.keys():
                    sim_role = key
                    break
                # appending the cost of living of obtained similar city
                col_col.append(city_col_data[sim_city])
                # Appending the average salary of obtained similar role
                target_data = avs_data[sim_city]
                avs_col.append(target_data[sim_role])
            # yield [75,100]
            print("Main part is completed!")
            # Adding the average salary column in raw dataframe
            raw_df["average salary per annum"] = avs_col
            # Adding cost of living data in raw dataframe
            raw_df["cost of living"] = col_col

            # applying function to convert the rates into integers
            raw_df["average salary per annum"] = raw_df["average salary per annum"].apply(conv_to_int)
            # applying function to calculate hourly rates
            raw_df["rph_feature"] = raw_df["average salary per annum"].apply(rate_calc)

            types = raw_df["campaign type"].unique()
            if len(types)>3:
                errors = f"""<h3>The model is trained on 3 campaign types(DTC,HCP and IC). <br>But in the uploaded file, there are {len(types)} campaign types. <br>  {', '.join(types).upper()} </h3>"""
                # yield [100,100]
                return
            # yield [80,100]
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
            # yield [85,100]
        except Exception as e:
            errors = f"""<h1 class="err"> Something is wrong with the data types of elements! <br> Please take this sample table as reference! </h1> <h4> Error: {e} </h4>"""
            # yield [100,100]
            return

        # Making final df for prediction
        final_df = raw_df[["class","role","rph_feature",'health care professional', 'integrated communications',"new york",'los angeles','cost of living']]
        # Let's have predictions
        predictions = model.predict(final_df)
        # Make a df out of predictions
        pred_df = pd.DataFrame({"Predicted Rates":predictions})
        pred_df["Predicted Rates"] = pred_df["Predicted Rates"].apply(lambda x: round(x))
        # Function to calculate margin
        def with_margin(x):
            wm = x + int((margin/100)*x)
            return wm
        # Applying the function to calculate margin and creating a new column our of it
        pred_df["Rates with Margin"] = pred_df["Predicted Rates"].apply(with_margin)
        # Concatenating the dataframes
        show_df = pd.concat((df,pred_df),axis=1)
        # yield [95,100]
        # Returning the dataframe to the user
        if filename.split(".")[-1]=="csv":
            filename="predicted"+filename
            show_df.to_csv(filename,index=False)
        elif filename.split(".")[-1]=="xlsx":
            filename="predicted"+filename
            show_df.to_excel(filename,index=False)
        html=show_df.head(10).to_html()
        # yield [100,100]
        t2 = threading.Thread(target=send_mail,args=(email,))
        t2.start()
        print("Completed!")
    except Exception as e:
            errors = f"""<h1 class="err"> Something is wrong with the file! <br> Please take this sample table as reference! </h1> <h4> Error: {e} </h4>"""
            # yield [100,100]
            return

def send_mail(email):
    import smtplib, ssl
    from email import encoders
    from email.mime.base import MIMEBase
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    smtp_server = "smtp.gmail.com"
    port = 587  # For starttls
    subject = "Hourly Rates Prediction!"
    body = """The following attachment contains the predicted outcomes.
            \nYou can download and see the results!
            """
    sender_email = "hourlyratesprediction@gmail.com"
    password = "thisishourlyratesprediction"
    receiver_email = email

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message["Bcc"] = receiver_email  # Recommended for mass emails

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    filepath = filename  # In same directory as script

    # Open PDF file in binary mode
    with open(filepath, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
    
        # Encode file in ASCII characters to send by email    
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {filename}",
    )

    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, port) as server:
        server.ehlo()  # Can be omitted
        server.starttls(context=context)
        server.ehlo()  # Can be omitted
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)

    print(f"Mail sent to {email}!")
    # global mail
    # msg = Message('Predicted Hourly Rates', sender = 'hourlyratesprediction@gmail.com', recipients = [email])
    # # msg.html = render_template('confirm.html', name=firstname, email=email) 
    # msg.html = "This is just a test! Please ignore this!"
    # mail.send(msg)
    pass

@app.route("/error")
def error():
    global errors
    sample_file = pd.ExcelFile("sample.xlsx")
    df = pd.read_excel(sample_file,"sample")
    sample_html = df.to_html()
    return render_template("upload.html",errors=errors,sample_file=sample_html)

@app.route("/successfull")
def successfull():
    global errors
    global html
    if not errors:
        return render_template("download.html",sample_file=html)
    else:
        return redirect(url_for('error'))    

# @app.route("/progress")
# def progress():

#     return Response(FPB.progress(process_file()), mimetype='text/event-stream')

@app.route("/submit", methods=['POST'])
def submit():
    if request.method=="POST":
        global filename
        global errors
        global margin
        file = request.files["file"]
        margin = int(request.form["Margin"])
        email = request.form["Email"]
        # email = "ysafarinep@gmail.com"
        # print(margin)
        # print(uploaded_file.filename)
        filename = secure_filename(file.filename)
        saving_path = "./uploads/" + filename
        file.save(saving_path)
        t1 = threading.Thread(target=process_file,args=(email,))
        t1.start()
        
        
        return render_template("progress.html")
    else:
        return render_template("upload.html")


@app.route("/predictions", methods=['POST'])
def predictions():
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



@app.route("/download-file")
def download():
    global filename
    return send_file(filename, attachment_filename=filename, as_attachment=True)

# Preprocessing the contents
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
    global index2word_set_w2v
    global num_features
    wv = avg_feature_vector(words,w2v_model,num_features,index2word_set_w2v)
    mod = np.linalg.norm(wv)
    return mod

# Similariy dictionary
def average_similarity_dict(given_role_city,mode):
    """Return the average similarity of sentence and all the keywords in file"""
    global index2word_set_cities
    global index2word_set_roles
    global rolecheck_model
    global citycheck_model
    # preprocessing of sentence and file
    sentence = preprocess(given_role_city)
    if mode=="c":
        with open("./files/cities.txt","r") as city_file:
            content = city_file.readlines()
        model = citycheck_model
        index2word_set = index2word_set_cities
    elif mode=="r":
        with open("./files/roles.txt","r") as roles_file:
            content = roles_file.readlines()
        model = rolecheck_model
        index2word_set = index2word_set_roles
    # print(content)
    # Removing the newline characters
    for index, value in enumerate(content):
        val = value.rstrip()
        content[index] = val
        total_sim = 0
        sim_dict = {}
        for role in content:
            s1_afv = avg_feature_vector(sentence, model=model, num_features=100, index2word_set=index2word_set)
            s2_afv = avg_feature_vector(role, model=model, num_features=100, index2word_set=index2word_set)
            sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
    # print(total_sim)
            sim_dict[role] = sim
    # sorting the dictionary
    new_dict = {}
    for wi in sorted(sim_dict, key=sim_dict.get, reverse=True):
        new_dict[wi] = sim_dict[wi]
    return new_dict

if __name__=="__main__":
    app.run(debug=True)


