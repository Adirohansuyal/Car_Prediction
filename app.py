from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the user
    try:
        ex_showroom_price = float(request.form['Ex-Showroom_Price'])
        cylinders = int(request.form['Cylinders'])
        valves_per_cylinder = int(request.form['Valves_Per_Cylinder'])
        cylinder_configuration = request.form['Cylinder_Configuration']
        
        # Prepare input for prediction (you might need to process this value depending on the model)
        input_data = np.array([[ex_showroom_price, cylinders, valves_per_cylinder, cylinder_configuration]])
        
        # Make prediction using the model
        prediction = model.predict(input_data)
        
        return jsonify({'car_name': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
