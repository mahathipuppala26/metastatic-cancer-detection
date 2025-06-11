from flask import Flask, request, jsonify
from PIL import Image
from predictor import predict_image
import os
import zipfile
import io

app = Flask(__name__)

def get_actual_label(filename):
    # Assuming filename contains '_label_1' for positive and else negative
    return "Cancerous" if "_label_1" in filename else "Non Cancerous"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get('file') or request.files.get('File')
    if file is None:
        return jsonify({"error": "No file part"}), 400

    # Check if uploaded file is a zip
    if not file.filename.endswith(".zip"):
        return jsonify({"error": "Please upload a zip file containing images"}), 400

    try:
        # Read zip file from memory
        in_memory_zip = io.BytesIO(file.read())
        with zipfile.ZipFile(in_memory_zip) as z:
            results = []
            vit_correct = 0
            vit_incorrect = 0
            cnn_correct = 0
            cnn_incorrect = 0

            for filename in z.namelist():
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    with z.open(filename) as image_file:
                        image = Image.open(image_file).convert("RGB")
                        predictions = predict_image(image)
                        actual = get_actual_label(filename)

                        # Count vit correctness
                        if predictions["vit_model_prediction"] == actual:
                            vit_correct += 1
                        else:
                            vit_incorrect += 1

                        # Count cnn correctness
                        if predictions["cnn_model_prediction"] == actual:
                            cnn_correct += 1
                        else:
                            cnn_incorrect += 1

                        results.append({
                            "filename": filename,
                            "actual_label": actual,
                            "vit_model_prediction": predictions["vit_model_prediction"],
                            "vit_model_confidence": round(predictions["vit_model_confidence"], 3),
                            "cnn_model_prediction": predictions["cnn_model_prediction"],
                            "cnn_model_confidence": round(predictions["cnn_model_confidence"], 3),
                        })

            summary = {
                "vit_model_correct": vit_correct,
                "vit_model_incorrect": vit_incorrect,
                "cnn_model_correct": cnn_correct,
                "cnn_model_incorrect": cnn_incorrect,
                "total_images": vit_correct + vit_incorrect
            }

            return jsonify({"results": results, "summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
