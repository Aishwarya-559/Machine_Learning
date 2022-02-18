from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import pickle

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        try:
            Pclass = int(request.form['Pclass'])
            Sex = int(request.form['Sex'])
            Age = int(request.form['Age'])
            SibSp = int(request.form['SibSp'])
            Parch = int(request.form['Parch'])
            Fare = int(request.form['Fare'])

            file = 'Decision_tree.pickle'
            loaded_model = pickle.load(open(file, 'rb'))
            prediction = loaded_model.predict([[Pclass, Sex, Age, SibSp, Parch, Fare]])
            print(f'Prediction is {prediction}')
            return render_template('result.html', prediction="Prediction of Survival is {} ".format(prediction))
        except Exception as e:
            return render_template('result.html', prediction='Prediction of Survival is {}'.format(e))

    else:
        return "Something went wrong"


if __name__ == '__main__':
    app.run(debug=True)
