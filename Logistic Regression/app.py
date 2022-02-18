from fileinput import filename
import fileinput
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle

app = Flask(__name__)

@app.route('/',methods = ['GET','POST'])
@cross_origin()

def index():
    return render_template('index.html')
filename = 'model.pkl'
@app.route('/predict', methods =['GET','POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        try:
            rate_marriage = int(request.form['rate_marriage'])
            age = int(request.form['age'])
            children = int(request.form['children'])
            religious = int(request.form['religious'])
            educ = int(request.form['educ'])
            occupation = int(request.form['occupation'])
            occupation_husb = int(request.form['occupation_husb'])
            

            loaded_model = pickle.load(open(filename, 'rb'))
            prediction = loaded_model.predict([[rate_marriage,age,children,religious,educ,occupation,occupation_husb]])
            print('Prediction is: ',prediction)
            return render_template('result.html', prediction= 'The Affair probability is {}'.format(prediction))
        except Exception as e:
            return render_template('result.html', prediction = 'The Affair probability is {}'.format(e))
    else:
        return "Something went wrong."
    
if __name__ == '__main__':
    app.run(debug=True)
        