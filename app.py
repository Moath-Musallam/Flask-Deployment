import numpy as np
from flask import Flask, request,render_template,jsonify
import pickle
import pandas as pd
# model for the web app

model = pickle.load(open('model.pkl', 'rb'))

# flask 

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

# web request

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
   
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output =prediction[0]
    if output ==0:
        output ='Iris-setosa'
    elif output ==1:
        output ='Iris-versicolor'
    elif output ==2:
        output ='Iris-virginica'

    return render_template('index.html', prediction_text='The Flower is {}'.format(output))


@app.route('/predict_API')
def predict_API():
    '''
    For rendering results on postman 
    '''
    model = pickle.load(open('model.pkl', 'rb'))
    sepal_length = request.args.get('sepal_length')
    sepal_width = request.args.get('sepal_width')
    petal_length = request.args.get('petal_length')
    petal_width = request.args.get('petal_width')
    test_df = pd.DataFrame({'sepal_length':[sepal_length],'sepal_width':[sepal_width], 'petal_length':[petal_length], 'petal_width':[petal_width]})
    
    pred_iris = model.predict(test_df).tolist()
    output =pred_iris[0]
    if output ==0:
        output ='Iris-setosa'
    elif output ==1:
        output ='Iris-versicolor'
    elif output ==2:
        output ='Iris-virginica'
    return jsonify({'pred_iris': output})

# if name = main the app will start 

if __name__ == '__main__':     
    app.run(debug=True)