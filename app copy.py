from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Cargar modelos y objetos de preprocesamiento
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
selector = joblib.load('selector.pkl')
svm_model = joblib.load('svm_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener datos del request
    data = request.get_json()

    # Verificar que los datos necesarios estén en el request
    required_fields = ['price', 'quantity_sold', 'customer_rating', 'review_count', 'category', 'store_location', 'discount_offered', 'customer_age_group', 'purchase_day', 'promotion_applied', 'payment_method', 'delivery_method']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400

    # Convertir datos a formato DataFrame
    data_df = pd.DataFrame([data])

    # Procesar datos
    categorical_columns = ['category', 'store_location', 'customer_age_group', 'purchase_day', 'payment_method', 'delivery_method']
    data_df[categorical_columns] = encoder.transform(data_df[categorical_columns])
    data_scaled = scaler.transform(data_df)
    data_optimal = pd.DataFrame(data_scaled).iloc[:, selector.support_]

    # Hacer predicción
    prediction = svm_model.predict(data_optimal)
    result = 'Popular' if prediction[0] else 'Not Popular'

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
