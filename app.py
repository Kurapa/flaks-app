from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import requests
from datetime import datetime
import os

# Direct Hugging Face URLs
SOIL_MODEL_URL = "https://huggingface.co/kuraeswar2005/soilmodel/resolve/main/soil_model.keras"
DISEASE_MODEL_URL = "https://huggingface.co/kuraeswar2005/plant_disease/resolve/main/plant_disease.keras"

# Local cache directory for models
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

try:
    print(f"⬇️ Streaming model from: {SOIL_MODEL_URL}")
    soil_model_path = tf.keras.utils.get_file(
        fname="soil_model.keras",
        origin=SOIL_MODEL_URL,
        cache_dir=MODEL_DIR,
        cache_subdir="."
    )

    print(f"⬇️ Streaming model from: {DISEASE_MODEL_URL}")
    disease_model_path = tf.keras.utils.get_file(
        fname="plant_disease.keras",
        origin=DISEASE_MODEL_URL,
        cache_dir=MODEL_DIR,
        cache_subdir="."
    )

    soil_model = tf.keras.models.load_model(soil_model_path)
    disease_model = tf.keras.models.load_model(disease_model_path)

    print("✅ Models loaded successfully!")

except Exception as e:
    print(f"❌ Error loading models: {e}")
    soil_model = None
    disease_model = None


# Soil type class labels
SOIL_CLASS_LABELS = {
    0: "alluvial", 1: "black", 2: "cinder", 3: "clay", 4: "laterite",
    5: "loamy", 6: "peat", 7: "red", 8: "sandy", 9: "sandy_loam", 10: "yellow"
}

# Disease class labels - UPDATE THESE BASED ON YOUR DISEASE MODEL
DISEASE_CLASS_LABELS = {
    0: "Pepper__bell___Bacterial_spot",
    1: "Pepper__bell___healthy",
    2: "Potato___Early_blight",
    3: "Potato___Late_blight",
    4: "Potato___healthy",
    5: "Tomato_Bacterial_spot",
    6: "Tomato_Early_blight",
    7: "Tomato_Late_blight",
    8: "Tomato_Leaf_Mold",
    9: "Tomato_Septoria_leaf_spot",
    10: "Tomato_Spider_mites_Two_spotted_spider_mite",
    11: "Tomato__Target_Spot",
    12: "Tomato__Tomato_YellowLeaf__Curl_Virus",
    13: "Tomato__Tomato_mosaic_virus",
    14: "Tomato_healthy"
}

# Plant type detection based on disease - UPDATE THIS MAPPING
DISEASE_TO_PLANT = {
    "Pepper__bell___Bacterial_spot": "Pepper bell",
    "Pepper__bell___healthy": "Pepper bell",
    "Potato___Early_blight": "Potato",
    "Potato___Late_blight": "Potato",
    "Potato___healthy": "Potato",
    "Tomato_Bacterial_spot": "Tomato",
    "Tomato_Early_blight": "Tomato",
    "Tomato_Late_blight": "Tomato",
    "Tomato_Leaf_Mold": "Tomato",
    "Tomato_Septoria_leaf_spot": "Tomato",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato",
    "Tomato__Target_Spot": "Tomato",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato",
    "Tomato__Tomato_mosaic_virus": "Tomato",
    "Tomato_healthy": "Tomato"
}

# Disease information database
DISEASE_INFO = {
    "Healthy": {
        "description": "The plant appears to be healthy with no visible signs of disease. Continue regular maintenance and preventive measures.",
        "symptoms": ["Vibrant green leaves", "No discoloration", "Normal growth patterns", "No visible lesions"],
        "causes": ["Good agricultural practices", "Proper nutrition", "Adequate water", "Disease-free seeds"],
        "severity": "low",
        "treatment": {
            "organic": ["Continue regular watering", "Maintain soil health with compost",
                        "Monitor regularly for any changes"],
            "chemical": ["No treatment needed"],
            "preventive": ["Maintain current practices", "Regular monitoring", "Crop rotation"]
        },
        "recommendedFertilizers": [
            {
                "name": "Balanced NPK Fertilizer",
                "type": "chemical",
                "dosage": "As per package instructions",
                "applicationMethod": "Soil application",
                "frequency": "Monthly",
                "price": "$10-15 per kg",
                "benefits": ["Maintains plant health", "Promotes growth", "Prevents deficiencies"]
            }
        ],
        "affectedArea": "0%",
        "estimatedLoss": "No yield loss expected",
        "actionRequired": "Continue regular maintenance and monitoring"
    },
    "Tomato Late Blight": {
        "description": "Late blight is a devastating disease caused by Phytophthora infestans. It can destroy entire crops within days if left untreated. The disease thrives in cool, wet conditions.",
        "symptoms": [
            "Dark brown to black lesions on leaves",
            "White fuzzy growth on undersides of leaves (in humid conditions)",
            "Rapid wilting and decay of foliage",
            "Brown, greasy-looking spots on fruits",
            "Characteristic musty odor"
        ],
        "causes": [
            "Cool, wet weather (15-20°C with high humidity)",
            "Poor air circulation between plants",
            "Overhead watering that keeps foliage wet",
            "Infected plant debris from previous season",
            "Wind-borne spores from nearby infected plants"
        ],
        "severity": "high",
        "treatment": {
            "organic": [
                "Remove and destroy infected plants immediately",
                "Apply copper-based fungicides as soon as symptoms appear",
                "Improve air circulation by pruning and proper spacing",
                "Water at the base of plants in the morning",
                "Use disease-free seeds and transplants"
            ],
            "chemical": [
                "Apply chlorothalonil fungicide at first sign of disease",
                "Use mancozeb as a preventive spray every 7-10 days",
                "Apply metalaxyl or mefenoxam for systemic protection",
                "Alternate fungicides to prevent resistance",
                "Follow label instructions carefully"
            ],
            "preventive": [
                "Plant resistant varieties when available",
                "Ensure proper spacing (60-90 cm between plants)",
                "Mulch to prevent soil splash onto leaves",
                "Remove lower leaves to improve air flow",
                "Rotate crops - avoid planting tomatoes in same location for 3-4 years",
                "Use drip irrigation instead of overhead sprinklers"
            ]
        },
        "recommendedFertilizers": [
            {
                "name": "Copper Oxychloride 50% WP",
                "type": "chemical",
                "dosage": "2-3 grams per liter of water",
                "applicationMethod": "Foliar spray covering both sides of leaves",
                "frequency": "Every 7-10 days until symptoms subside",
                "price": "$15-20 per kg",
                "benefits": ["Prevents fungal spore germination", "Protects new growth",
                             "Long-lasting protective effect"]
            },
            {
                "name": "Neem Oil Solution (Organic)",
                "type": "organic",
                "dosage": "5ml per liter of water",
                "applicationMethod": "Spray thoroughly on all plant parts",
                "frequency": "Every 5-7 days as preventive measure",
                "price": "$10-12 per liter",
                "benefits": ["100% organic and safe", "No chemical residue", "Also controls other pests",
                             "Safe for beneficial insects"]
            },
            {
                "name": "Potassium Phosphonate",
                "type": "bio",
                "dosage": "2ml per liter of water",
                "applicationMethod": "Foliar spray and soil drench",
                "frequency": "Every 2 weeks for prevention",
                "price": "$25-30 per liter",
                "benefits": ["Systemic action throughout plant", "Boosts plant immunity",
                             "Rapid uptake and translocation", "No resistance development"]
            }
        ],
        "affectedArea": "30-60% of field typically affected",
        "estimatedLoss": "Potential 50-100% yield loss if left untreated within 1-2 weeks",
        "actionRequired": "URGENT: Immediate treatment required within 24-48 hours to prevent total crop loss"
    },
    "Tomato Early Blight": {
        "description": "Early blight is a fungal disease caused by Alternaria solani. It typically affects older, lower leaves first and can significantly reduce yield if not managed properly.",
        "symptoms": [
            "Brown spots with concentric rings (target-like or bull's-eye pattern)",
            "Yellowing around the lesions",
            "Progressive leaf drop starting from bottom of plant",
            "Dark lesions on stems near soil line",
            "Spots on fruit with dark, leathery sunken areas"
        ],
        "causes": [
            "Warm temperatures (24-29°C) with high humidity",
            "Poor air circulation and dense planting",
            "Overhead irrigation that wets foliage",
            "Stressed plants due to nutrient deficiency",
            "Contaminated soil from previous crops"
        ],
        "severity": "medium",
        "treatment": {
            "organic": [
                "Remove and destroy affected leaves promptly",
                "Apply organic fungicides containing Bacillus subtilis",
                "Mulch around plants to prevent soil splash",
                "Improve plant nutrition with balanced fertilizer",
                "Prune lower leaves to improve air circulation"
            ],
            "chemical": [
                "Apply chlorothalonil fungicide every 7-10 days",
                "Use mancozeb as protective spray",
                "Apply azoxystrobin for systemic protection",
                "Alternate between different fungicide groups",
                "Start applications before symptoms appear in high-risk conditions"
            ],
            "preventive": [
                "Plant disease-resistant varieties",
                "Stake plants to keep foliage off ground",
                "Water at base of plants, avoid wetting leaves",
                "Practice 3-4 year crop rotation",
                "Remove plant debris at end of season",
                "Maintain adequate plant spacing (60-75 cm)"
            ]
        },
        "recommendedFertilizers": [
            {
                "name": "Chlorothalonil 75% WP",
                "type": "chemical",
                "dosage": "2 grams per liter of water",
                "applicationMethod": "Thorough foliar spray",
                "frequency": "Every 7-10 days during disease pressure",
                "price": "$18-22 per kg",
                "benefits": ["Broad-spectrum protection", "Prevents spore germination", "Cost-effective",
                             "Long residual activity"]
            },
            {
                "name": "Mancozeb 75% WG",
                "type": "chemical",
                "dosage": "2 grams per liter of water",
                "applicationMethod": "Foliar spray",
                "frequency": "Every 7-14 days as preventive",
                "price": "$12-16 per kg",
                "benefits": ["Multi-site fungicide", "Low resistance risk", "Effective protectant", "Economical option"]
            },
            {
                "name": "Copper-Zinc Foliar Spray",
                "type": "organic",
                "dosage": "1-2 ml per liter",
                "applicationMethod": "Foliar application",
                "frequency": "Bi-weekly application",
                "price": "$15-20 per liter",
                "benefits": ["Organic option", "Strengthens plant immunity", "Micronutrient supplement",
                             "Reduces disease severity"]
            }
        ],
        "affectedArea": "20-40% of foliage typically affected",
        "estimatedLoss": "Potential 20-50% yield reduction if unmanaged",
        "actionRequired": "Treatment recommended within 3-5 days to prevent yield loss"
    },
    "Wheat Rust": {
        "description": "Wheat rust is a fungal disease that appears as orange-brown pustules on leaves and stems. It can spread rapidly and significantly impact grain quality and yield.",
        "symptoms": [
            "Orange to reddish-brown pustules on leaves",
            "Yellow spots or halos around pustules",
            "Powdery spores that rub off on hands",
            "Premature leaf yellowing and death",
            "Reduced grain filling and shriveled kernels"
        ],
        "causes": [
            "Warm days (20-25°C) with cool nights",
            "High humidity and dew formation",
            "Dense crop stands with poor air movement",
            "Excess nitrogen fertilization",
            "Wind-borne spores from infected fields"
        ],
        "severity": "medium",
        "treatment": {
            "organic": [
                "Use sulfur-based organic fungicides",
                "Remove infected plant debris",
                "Increase plant spacing in future plantings",
                "Apply compost tea to boost plant immunity"
            ],
            "chemical": [
                "Apply triazole fungicides (propiconazole, tebuconazole)",
                "Use strobilurin fungicides for systemic action",
                "Apply at flag leaf stage for best protection",
                "Consider tank mixing for broader spectrum control"
            ],
            "preventive": [
                "Plant rust-resistant wheat varieties",
                "Adjust planting dates to avoid peak infection periods",
                "Manage nitrogen fertilization properly",
                "Scout fields regularly during susceptible stages",
                "Remove volunteer wheat plants that harbor rust"
            ]
        },
        "recommendedFertilizers": [
            {
                "name": "Propiconazole 25% EC",
                "type": "chemical",
                "dosage": "1ml per liter of water",
                "applicationMethod": "Foliar spray",
                "frequency": "Every 14-21 days if needed",
                "price": "$35-45 per liter",
                "benefits": ["Systemic protection", "Long residual effect", "Curative and preventive action",
                             "Effective on multiple rust types"]
            },
            {
                "name": "Tebuconazole 25% EC",
                "type": "chemical",
                "dosage": "0.5-1ml per liter",
                "applicationMethod": "Foliar spray at boot to heading stage",
                "frequency": "One to two applications per season",
                "price": "$40-50 per liter",
                "benefits": ["Excellent systemic activity", "Protects new growth", "Improves grain quality",
                             "Long-lasting protection"]
            },
            {
                "name": "Sulfur 80% WP (Organic)",
                "type": "organic",
                "dosage": "3 grams per liter",
                "applicationMethod": "Foliar spray",
                "frequency": "Every 10-14 days",
                "price": "$8-12 per kg",
                "benefits": ["Organic certified", "No resistance issues", "Also provides sulfur nutrition",
                             "Safe for environment"]
            }
        ],
        "affectedArea": "15-30% of field typically affected",
        "estimatedLoss": "Potential 15-40% yield loss depending on timing and severity",
        "actionRequired": "Treatment recommended within 1 week of detection"
    }
}

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration


# Helper function to get severity based on confidence
def get_severity(confidence, disease_name):
    if disease_name == "Healthy":
        return "low"

    # Get base severity from disease info
    base_severity = DISEASE_INFO.get(disease_name, {}).get("severity", "medium")

    # Adjust based on confidence
    if confidence < 70:
        return "medium"
    elif confidence >= 70 and confidence < 85:
        return base_severity
    else:
        # High confidence - return the base severity
        if base_severity == "medium" and confidence > 90:
            return "high"
        return base_severity


# Helper function to calculate affected area
def calculate_affected_area(confidence, severity):
    if severity == "low":
        return "0-5%"
    elif severity == "medium":
        base = 15 + (confidence - 70) * 0.5
        return f"{int(base)}-{int(base + 15)}%"
    else:  # high or critical
        base = 30 + (confidence - 80) * 1.0
        return f"{int(base)}-{int(base + 30)}%"


@app.route('/')
def home():
    return jsonify({
        "message": "Agricultural AI API",
        "endpoints": {
            "/predict": "Soil type prediction",
            "/api/detect-disease": "Plant disease detection"
        },
        "status": "online"
    })


@app.route('/predict', methods=['POST'])
def predict_soil():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        image = Image.open(io.BytesIO(file.read()))

        # Preprocess the image
        image = image.resize((128, 128))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Make prediction
        predictions = soil_model.predict(image)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        return jsonify({
            "soil_type": SOIL_CLASS_LABELS.get(predicted_class, "Unknown"),
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/detect-disease', methods=['POST'])
def detect_disease():
    """
    Endpoint for plant disease detection
    Accepts: multipart/form-data with 'image' file or 'imageUrl' string
    Returns: Complete disease analysis result
    """
    try:
        image = None
        print(request.form['imageUrl'])
        # Try to get image from file upload
        if 'image' in request.files:
            file = request.files['image']
            image = Image.open(io.BytesIO(file.read()))

        # Try to get image from URL
        elif 'imageUrl' in request.form:
            image_url = request.form['imageUrl']
            response = requests.get(image_url)
            image = Image.open(io.BytesIO(response.content))

        else:
            return jsonify({"error": "No image provided. Send 'image' file or 'imageUrl'"}), 400

        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Preprocess the image (adjust size based on your model)
        original_size = image.size
        image_resized = image.resize((224, 224))  # Common size for plant disease models
        image_array = np.array(image_resized) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Make prediction
        predictions = disease_model.predict(image_array)
        predicted_class_idx = np.argmax(predictions[0])
        print(predicted_class_idx)
        confidence = float(np.max(predictions[0])) * 100

        # Get disease name
        disease_name = DISEASE_CLASS_LABELS.get(predicted_class_idx, "Unknown Disease")
        print(disease_name)
        # Get plant type
        plant_type = DISEASE_TO_PLANT.get(disease_name, "Unknown Plant")

        # Get disease information
        disease_data = DISEASE_INFO.get(disease_name, DISEASE_INFO.get("Healthy"))

        # Calculate severity based on confidence and disease type
        severity = get_severity(confidence, disease_name)

        # Calculate affected area
        affected_area = calculate_affected_area(confidence, severity)

        # Build response according to frontend DiseaseResult interface
        result = {
            "disease": disease_name,
            "plantType": plant_type,
            "confidence": round(confidence, 2),
            "severity": severity,
            "description": disease_data.get("description", "No description available"),
            "symptoms": disease_data.get("symptoms", []),
            "causes": disease_data.get("causes", []),
            "treatment": disease_data.get("treatment", {
                "organic": [],
                "chemical": [],
                "preventive": []
            }),
            "recommendedFertilizers": disease_data.get("recommendedFertilizers", []),
            "affectedArea": affected_area,
            "estimatedLoss": disease_data.get("estimatedLoss", "Unable to estimate"),
            "actionRequired": disease_data.get("actionRequired", "Monitor closely and consult agricultural expert"),
            "metadata": {
                "originalImageSize": f"{original_size[0]}x{original_size[1]}",
                "modelInputSize": "224x224",
                "predictionTime": datetime.now().isoformat(),
                "allPredictions": {
                    DISEASE_CLASS_LABELS.get(i, f"Class_{i}"): round(float(predictions[0][i]) * 100, 2)
                    for i in range(len(predictions[0]))
                }
            }
        }

        return jsonify(result), 200

    except Exception as e:
        print(f"Error in disease detection: {str(e)}")
        return jsonify({
            "error": "Disease detection failed",
            "details": str(e)
        }), 500


@app.route('/api/disease-info/<disease_name>', methods=['GET'])
def get_disease_info(disease_name):
    """
    Get detailed information about a specific disease
    """
    try:
        # URL decode the disease name
        import urllib.parse
        disease_name = urllib.parse.unquote(disease_name)

        if disease_name in DISEASE_INFO:
            return jsonify(DISEASE_INFO[disease_name]), 200
        else:
            return jsonify({"error": "Disease information not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/diseases', methods=['GET'])
def get_all_diseases():
    """
    Get list of all diseases in the database
    """
    try:
        diseases = [
            {
                "name": disease_name,
                "severity": info.get("severity", "unknown"),
                "description": info.get("description", "")[:100] + "..."
            }
            for disease_name, info in DISEASE_INFO.items()
        ]
        return jsonify({"diseases": diseases, "count": len(diseases)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Check if the API and models are working
    """
    try:
        return jsonify({
            "status": "healthy",
            "soil_model": "loaded" if soil_model else "not loaded",
            "disease_model": "loaded" if disease_model else "not loaded",
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


if __name__ == '__main__':
    print("=" * 50)
    print("Agricultural AI API Server Starting...")
    print("=" * 50)
    print(f"Soil Model: {'✓ Loaded' if soil_model else '✗ Failed'}")
    print(f"Disease Model: {'✓ Loaded' if disease_model else '✗ Failed'}")
    print(f"Available Endpoints:")
    print("  - POST /predict (Soil Type)")
    print("  - POST /api/detect-disease (Disease Detection)")
    print("  - GET  /api/disease-info/<name>")
    print("  - GET  /api/diseases")
    print("  - GET  /api/health")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
