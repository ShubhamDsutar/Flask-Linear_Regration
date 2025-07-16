# from flask import Flask, request, render_template 
# import pickle 
# import numpy as np 
 
# app = Flask(__name__) 
 
# # Load the trained model 
# model = pickle.load(open('linear_model.pkl', 'rb')) 
 
# @app.route('/') 
# def home(): 
#     return render_template('index11.html') 
 
# @app.route('/predict', methods=['POST']) 
# def predict(): 
#     try: 
#         x = float(request.form['x'])  # Change as per your model inputs 
        
 
#         features = np.array([[x]]) 
#         prediction = model.predict(features) 
 
#         return render_template('index11.html', result=f'Predicted Value: {prediction[0]:.2f}') 
#     except Exception as e: 
#         return render_template('index11.html', result=f'Error: {str(e)}') 
 
# if __name__ == '__main__': 
#     app.run(debug=True)
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('linear_model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', result='')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        x = float(request.form['x'])  # Get user input
        features = np.array([[x]])    # Reshape for model
        prediction = model.predict(features)  # Predict
        return render_template('index.html', result=f'Predicted Value: ₹{prediction[0]:,.2f}')
    except Exception as e:
        return render_template('index.html', result=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
