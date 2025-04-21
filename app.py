from flask import Flask, request, jsonify
from training_model import load_or_train_model, predict_component

# Load model saat startup
model = load_or_train_model()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        description = data['description']
        predictions = predict_component(model, description)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
