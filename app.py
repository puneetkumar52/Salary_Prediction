from flask import Flask, request, render_template
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

app = Flask(__name__)

# Sample data — replace this with your real dataset if needed
data = pd.DataFrame({
    'YearsExperience': [1.1, 2.0, 3.2, 4.5, 5.5, 6.9, 7.9],
    'Salary': [39343, 45000, 60000, 61000, 83000, 93940, 98273]
})

X = data[['YearsExperience']]
y = data['Salary']

# Train using Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        experience = float(request.form['experience'])
        prediction = model.predict(np.array([[experience]]))
        output = round(prediction[0], 2)
        return render_template('index.html', prediction_text=f"Predicted Salary: ₹ {output:,.2f}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
