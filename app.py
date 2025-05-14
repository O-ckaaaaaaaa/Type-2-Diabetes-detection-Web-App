from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import tensorflow as tf
import numpy as np
import joblib
from flask_sqlalchemy import SQLAlchemy
from flask import send_file
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import io
import base64
from flask import Response
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Initialize database
def init_db():
    with sqlite3.connect('database.db') as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            )
        ''')

init_db()

app = Flask(__name__)
app.secret_key = "your_secret_key"
# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load trained model and scaler
model = tf.keras.models.load_model('diabetes_prediction_model.h5')
scaler = joblib.load('scaler.pkl')
# Database Model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pregnancies = db.Column(db.Integer, nullable=False)
    glucose = db.Column(db.Integer, nullable=False)
    blood_pressure = db.Column(db.Integer, nullable=False)
    skin_thickness = db.Column(db.Integer, nullable=False)
    insulin = db.Column(db.Integer, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    dpf = db.Column(db.Float, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    probability = db.Column(db.Float, nullable=False)
    prediction = db.Column(db.Integer, nullable=False)

# Home
@app.route('/')
def home():
    return redirect(url_for('login'))

# Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])

        with sqlite3.connect('database.db') as conn:
            try:
                conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                return "Username already exists."

    return render_template('register.html')

# Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        with sqlite3.connect('database.db') as conn:
            cur = conn.cursor()
            cur.execute('SELECT * FROM users WHERE username = ?', (username,))
            user = cur.fetchone()

            if user and check_password_hash(user[2], password):
                session['user_id'] = user[0]
                session['username'] = user[1]
                return redirect(url_for('index'))
            else:
                return "Invalid username or password."

    return render_template('login.html')

# Dashboard (Protected Route)
@app.route('/index')
def index():
    if 'user_id' in session:
       predictions = Prediction.query.all()
       latest_prediction = session.get('latest_prediction', None)  # Retrieve latest from session
       return render_template('index.html',latest_prediction=latest_prediction,predictions=predictions, username=session['username'])
    return redirect(url_for('login'))
"""
@app.route('/')
def index():
    predictions = Prediction.query.all()
    latest_prediction = session.get('latest_prediction', None)  # Retrieve latest from session
    return render_template('index.html', latest_prediction=latest_prediction, predictions=predictions)
"""
# Logout
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array([data['features']])
        features_scaled = scaler.transform(features)

        # Predict using the model
        prediction_prob = model.predict(features_scaled)[0][0]
        prediction = 1 if prediction_prob > 0.5 else 0

        # Generate health recommendations
        recommendations = get_health_recommendations(prediction, features[0])

        # Store in database
        new_prediction = Prediction(
            pregnancies=data['features'][0],
            glucose=data['features'][1],
            blood_pressure=data['features'][2],
            skin_thickness=data['features'][3],
            insulin=data['features'][4],
            bmi=data['features'][5],
            dpf=data['features'][6],
            age=data['features'][7],
            probability=prediction_prob,
            prediction=prediction
        )
        db.session.add(new_prediction)
        db.session.commit()

        # Store latest prediction in session
        session['latest_prediction'] = {
            "id": new_prediction.id,
            "pregnancies": new_prediction.pregnancies,
            "glucose": new_prediction.glucose,
            "blood_pressure": new_prediction.blood_pressure,
            "skin_thickness": new_prediction.skin_thickness,
            "insulin": new_prediction.insulin,
            "bmi": new_prediction.bmi,
            "dpf": new_prediction.dpf,
            "age": new_prediction.age,
            "probability": new_prediction.probability,
            "prediction": new_prediction.prediction,
            "recommendations": recommendations
        }
        return jsonify({
            "prediction": "High Risk" if prediction == 1 else "Low Risk",
            "probability": prediction_prob,
            "recommendations": recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predictions', methods=['GET'])
def get_predictions():
        predictions = Prediction.query.all()
        results = [{
            "id": p.id,
            "pregnancies": p.pregnancies,
            "glucose": p.glucose,
            "blood_pressure": p.blood_pressure,
            "skin_thickness": p.skin_thickness,
            "insulin": p.insulin,
            "bmi": p.bmi,
            "dpf": p.dpf,
            "age": p.age,
            "probability": p.probability,
            "prediction": p.prediction
        } for p in predictions]
        return jsonify(results)

@app.route('/predictions_page', methods=['GET'])
def predictions_page():
    predictions = Prediction.query.all()
    latest_prediction = session.get('latest_prediction', None)  # Retrieve from session
    return render_template('predictions.html', predictions=predictions, latest_prediction=latest_prediction)

# Health Recommendations Generator
def get_health_recommendations(prediction, features):
    recommendations = []
    if prediction == 1:
        recommendations.append("You are at high risk of diabetes. Take the following steps:")
        if features[1] > 140:
            recommendations.append("- Monitor your blood sugar regularly.")
        if features[5] > 30:
            recommendations.append("- Maintain a healthy weight through diet and exercise.")
        if features[2] > 90:
            recommendations.append("- Monitor your blood pressure regularly.")
        recommendations.append("- Consult a healthcare professional for a full diagnosis.")
    else:
        recommendations.append("Your risk of diabetes is low, but stay healthy by:")
        if features[5] > 25:
            recommendations.append("- Maintaining a balanced diet and active lifestyle.")
        recommendations.append("- Regularly checking blood sugar levels.")
    return "\n".join(recommendations)

@app.route('/delete_prediction/<int:prediction_id>', methods=['POST'])
def delete_prediction(prediction_id):
    try:
        prediction_to_delete = Prediction.query.get(prediction_id)
        if prediction_to_delete:
            db.session.delete(prediction_to_delete)
            db.session.commit()
            return redirect(url_for('predictions_page'))  # Redirect after deletion
        else:
            return jsonify({'error': 'Prediction not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/download_report')
def download_report():
            buffer = io.BytesIO()

            # Create PDF document
            pdf = canvas.Canvas(buffer, pagesize=letter)
            pdf.setTitle("Diabetes Prediction Report")

            # Title
            pdf.setFont("Helvetica-Bold", 16)
            pdf.drawString(200, 750, "Diabetes Prediction Report")

            # Subtitle
            pdf.setFont("Helvetica", 12)
            pdf.drawString(200, 730, "Generated from your recent predictions")

            # Sample Data
            predictions = Prediction.query.all()  # Fetch all predictions from DB

            y_position = 700
            pdf.setFont("Helvetica-Bold", 10)
            pdf.drawString(50, y_position, "ID | Glucose | BMI | Probability | Risk Level")
            pdf.setFont("Helvetica", 10)

            y_position -= 20
            for p in predictions[:10]:  # Limit to 10 records for clarity
                risk_level = "High" if p.prediction == 1 else "Low"
                pdf.drawString(50, y_position,
                               f"{p.id} | {p.glucose} | {p.bmi} | {round(p.probability * 100, 2)}% | {risk_level}")
                y_position -= 20
                if y_position < 100:
                    pdf.showPage()
                    y_position = 750

            # Save PDF
            pdf.showPage()
            pdf.save()
            buffer.seek(0)

            return send_file(buffer, as_attachment=True, download_name="Diabetes_Report.pdf",
                             mimetype="application/pdf")

@app.route('/visualization')
def generate_graphs():
    predictions = Prediction.query.all()

    # Convert query data to Pandas DataFrame
    data = pd.DataFrame([{
        "glucose": p.glucose,
        "bmi": p.bmi,
        "probability": p.probability,
        "prediction": p.prediction
    } for p in predictions])

    if data.empty:
        return "No data available for visualization"

    # Create a plot (Glucose vs Probability)
    fig, ax = plt.subplots()
    ax.scatter(data['glucose'], data['probability'], c=data['prediction'], cmap='coolwarm', alpha=0.7)
    ax.set_title("Glucose vs. Diabetes Risk")
    ax.set_xlabel("Glucose Level")
    ax.set_ylabel("Diabetes Probability")

    # Convert to an image response
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return Response(img.getvalue(), mimetype='image/png')
if __name__ == '__main__':
    app.run(debug=True)
