from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)

# Configurar CORS
CORS(app, origins=["https://austins.vercel.app"])  # Reemplaza con tu dominio real en Vercel

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
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({'error': f'Missing fields: {", ".join(missing_fields)}'}), 400

    # Convertir datos a formato DataFrame
    data_df = pd.DataFrame([data])

    # Procesar datos
    categorical_columns = ['category', 'store_location', 'customer_age_group', 'purchase_day', 'payment_method', 'delivery_method']
    try:
        data_df[categorical_columns] = encoder.transform(data_df[categorical_columns])
        data_scaled = scaler.transform(data_df)
        data_optimal = pd.DataFrame(data_scaled).iloc[:, selector.support_]
    except Exception as e:
        return jsonify({'error': f'Error in processing data: {str(e)}'}), 400

    # Hacer predicción
    try:
        prediction = svm_model.predict(data_optimal)
        result = 'Popular' if prediction[0] else 'Not Popular'
    except Exception as e:
        return jsonify({'error': f'Error in prediction: {str(e)}'}), 500

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
