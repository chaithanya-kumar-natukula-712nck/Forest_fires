from flask import Flask,request,jsonify,render_template
import pickle
application  = Flask(__name__)

app =application

ridge_model=pickle.load(open('models/ridge.pkl','rb'))
Standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/home")
def home():
    return render_template("home.html")

@app.route('/predict_datapoint',methods=["GET","POST"])
def predict_data():
    if request.method=="POST":
        temperature = float(request.form['Temperature'])
        humidity = float(request.form['RH'])
        wind = float(request.form['Ws'])
        rain = float(request.form['Rain'])
        ffmc = float(request.form['FFMC'])
        dmc = float(request.form['DMC'])
        isi = float(request.form['ISI'])
        region = int(request.form['Region'])


        new_data_scaled = Standard_scaler.transform(
        [[temperature, humidity, wind, rain, ffmc, dmc, isi, region]]
    )
        
        result=ridge_model.predict(new_data_scaled)
        return render_template('home.html',results=result[0])

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)