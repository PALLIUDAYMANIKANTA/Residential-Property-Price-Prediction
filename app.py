from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and encoders

with open('/home/UdayManikanta/Project/Etree_regmodel.pkl', 'rb') as file:
    model = pickle.load(file)

with open('/home/UdayManikanta/Project/encoders.pkl', 'rb') as file:
    encoders = pickle.load(file)

# Define the manual encoding mappings
status_encoding = {
    'Under Construction': 3,
    'Ready to move': 1,
    'New': 0,
    'Resale': 2
}

@app.route('/', methods = ['GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    location = request.form['Location']
    area = float(request.form['Area'])
    prc_sqft = float(request.form['Prc_Sqft'])
    status = request.form['Status']
    bathrooms = request.form['Bathrooms']

    # Perform manual encoding on the input values
    status_encoded = status_encoding.get(status)
    bathrooms_encoded = encoders['Bathrooms'].transform([bathrooms])[0]
    location_encoded = encoders['Location'].transform([location])[0]

    # Create the input array for prediction
    input_data = np.array([location_encoded, area, prc_sqft, status_encoded, bathrooms_encoded])

    # Make the prediction
    prediction = model.predict([input_data])[0]

    # Render the template with the prediction result
    return render_template('home.html', prediction_text='The predicted price is: {}'.format(prediction))

