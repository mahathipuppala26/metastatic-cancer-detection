from flask import Flask, request, jsonify
from PIL import Image
from predictor2 import predict_image
import os

app = Flask(__name__)

def get_actual_label(filename):
    return "Cancerous" if "_label_1" in filename else "Non Cancerous"

@app.route("/predict_single", methods=["POST"])
def predict_single():
    file = request.files.get('file') or request.files.get('File')

    if file is None:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        filename = file.filename
        actual_label = get_actual_label(filename)

        image = Image.open(file.stream).convert("RGB")
        predictions = predict_image(image)

        return jsonify({
            "filename": filename,
            "actual_label": actual_label,
            "vit_model_prediction": predictions["vit_model_prediction"],
            "cnn_model_prediction": predictions["cnn_model_prediction"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
