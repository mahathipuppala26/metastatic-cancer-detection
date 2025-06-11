#Metastatic Cancer Detection using Vision Transformers (ViT)

This project presents a deep learning pipeline to automatically detect metastatic cancer from histopathology images using a Vision Transformer (ViT) model, with comparison to a Convolutional Neural Network (CNN) baseline. Both models are trained on the PatchCamelyon (PCam) dataset and tested using a Flask API deployed locally and integrated with Postman for real-time inference.

| Notebook                  | Description |
|---------------------------|-------------|
| `vit-model.ipynb`   | Trains a full ViT from scratch with patch extraction, positional encoding, and attention blocks |
| `cnn-model.ipynb`               | Builds and trains a simple 3-layer CNN for baseline comparison |
| `testing-vit.ipynb`       | Loads trained ViT model and performs prediction on new test images |

## 🔌 Flask API:

A lightweight API was created using Flask to simulate real-time predictions.

### `app.py` — ZIP Upload Endpoint

- Accepts `.zip` file containing multiple `.jpg/.png` histopathology images
- Extracts and preprocesses each image
- Returns predictions using ViT model (`vit_model.h5`)
- Outputs a clean JSON response with:
  - filename
  - predicted label (cancerous / non-cancerous)
  - confidence score

### `predictor.py`
- Handles loading the model, image preprocessing, and prediction

Similarly the same for testing a single image which is done using app2.py and predictor2.py.

###  `requirements.txt`

Install Flask and model dependencies:

```bash
pip install -r requirements.txt



***Testing the API with Postman:***

1. Start the API:
python app.py
Flask server starts at: http://localhost:5000

2. Postman Setup
Method: POST

URL: http://localhost:5000/predict

Body → form-data: Key	Value	Type
file	your_images.zip	File

For single image testing:
Go with app2.py and give single image in postman POST method's url.

you get example response in json format:

{
  "img_1.png": {
    "prediction": "cancerous",
    "confidence": 0.941
  },
  "img_2.jpg": {
    "prediction": "non-cancerous",
    "confidence": 0.883
  }
}

