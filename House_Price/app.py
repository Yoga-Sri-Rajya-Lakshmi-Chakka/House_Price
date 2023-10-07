from flask import Flask, render_template, request
import numpy as np
import pickle
app=Flask(__name__)
with open('House.pkl','rb') as f:
    model=pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    bed=int(request.form['bedrooms'])
    bath=int(request.form['bathrooms'])
    location=int(request.form['location'])
    size=int(request.form['area'])
    status=int(request.form['status'])
    facing=int(request.form['facing'])
    Type=int(request.form['type'])

    data=np.array([[bed,bath,location,size,status,facing,Type]])

    prediction=model.predict(data)[0]

    return render_template('index.html',prediction=prediction)

if __name__=='__main__':
    app.run(debug=True)