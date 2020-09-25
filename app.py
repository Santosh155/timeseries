from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('final_year_pharmacy_forecasting_project', 'rb'))




@app.route('/')
def hello_world():
    return render_template("pharmacy.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    item = request.form['item']
    if item == 'Paracetamol' or item == 'paracetamol':
        item = 'Item-1'
    elif item == 'Sinex' or item == 'sinex':
        item = 'Item-2'
    else:
        pass
    start = request.form['start']
    end = request.form['end']
    output = model[item].predict(start=start, end=end)
    output = output.astype(np.int64)
    frame = output.to_frame()
    # rename = output.rename(columns={'predicted_mean': 'Stock Forecast'})
    return render_template('pharmacy.html', pred='{}'.format(frame))


if __name__ == '__main__':
    app.run(debug=True)