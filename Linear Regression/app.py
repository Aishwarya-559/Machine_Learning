from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle

app = Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
@cross_origin()
def predict():
    if request.method == 'POST':
        try:
            CRIM = float(request.form['CRIM'])
            RM = float(request.form['RM'])
            AGE = float(request.form['AGE'])
            RAD = float(request.form['RAD'])
            LSTAT = float(request.form['LSTAT'])

            filename = 'model.pkl'
            loaded_model = pickle.load(open(filename,'rb'))
            prediction = loaded_model.predict([['CRIM','RM','AGE','RAD','LSTAT']])
            print('Prediction is: ',prediction)
            return
            render_template('results.html')
        except Exception as e:
            print('The exception message is:', e)
            return render_template('result.html')
    else:
        return 'Something went wrong'

if __name__ == '__main__':
    app.run(debug=True)