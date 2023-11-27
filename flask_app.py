import subprocess
import sys
from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template
import pickle
import pandas as pd


# from flask_cors import CORS
# try:
#     from flask_cors import CORS
# except:
#     subprocess.check_call([sys.executable,'-m','pip','install','flask_cors']);

# from flask import Flask, render_template, request, jsonify
# from flask_cors import CORS, cross_origin
# import pandas as pd
# import pickle

# app = Flask(__name__)

# @app.route('/')
# def hello_world():
#     return 'Hello from Flask!'



# load the pre-trained model
# with open('dtmodel.pkl', 'rb') as f:
#     model = pickle.load(f)

# model_filename = 'dtmodel.pkl'  # Replace with the actual model file name
# print(f"Attempting to load {model_filename} from: {os.path.abspath(model_filename)}")

# try:
#     with open(model_filename, 'rb') as f:
#         model = pickle.load(f)
# except Exception as e:
#     print(f"Error loading the model: {e}")


try:
    with open('dtmodel.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading the model: {e}")

try:
    with open('encoderCrop.pkl', 'rb') as f:
        encoder = pickle.load(f)
except Exception as e:
    print(f"Error loading the model: {e}")

try:
    with open('encoderDistrict.pkl', 'rb') as f:
        district_loaded = pickle.load(f)
except Exception as e:
    print(f"Error loading the model: {e}")

try:
    with open('ampalayaModel.pkl', 'rb') as f:
        ampalayamodel = pickle.load(f)
except Exception as e:
    print(f"Error loading the model: {e}")

try:
    with open('kmModel.pkl', 'rb') as f:
        kamatismodel = pickle.load(f)
except Exception as e:
    print(f"Error loading the model: {e}")

try:
    with open('cassavaModel.pkl', 'rb') as f:
        cassavamodel = pickle.load(f)
except Exception as e:
    print(f"Error loading the model: {e}")

try:
    with open('maisModel.pkl', 'rb') as f:
        maismodel = pickle.load(f)
except Exception as e:
    print(f"Error loading the model: {e}")

try:
    with open('palayModel.pkl', 'rb') as f:
        palaymodel = pickle.load(f)
except Exception as e:
    print(f"Error loading the model: {e}")

try:
    with open('talongModel.pkl', 'rb') as f:
        talongmodel = pickle.load(f)
except Exception as e:
    print(f"Error loading the model: {e}")

try:
    with open('patatasModel.pkl', 'rb') as f:
        patatasmodel = pickle.load(f)
except Exception as e:
    print(f"Error loading the model: {e}")

try:
    with open('enLT.pkl', 'rb') as f:
        landType_loaded = pickle.load(f)
except Exception as e:
    print(f"Error loading the model: {e}")

try:
    with open('enST.pkl', 'rb') as f:
        seedType_loaded = pickle.load(f)
except Exception as e:
    print(f"Error loading the model: {e}")


# create a Flask app
app = Flask(__name__)
# CORS(app)

# CORS(app, resources={r"/Agricrop": {"origins": "http://localhost:5002"}})
# CORS(app, resources={r"/PredictCrop": {"origins": "http://localhost:5002"}})
# CORS(app, resources={r"/PredictCropKamatis": {"origins": "http://localhost:5002"}})
# CORS(app, resources={r"/PredictCropCassava": {"origins": "http://localhost:5002"}})
# CORS(app, resources={r"/PredictCropMais": {"origins": "http://localhost:5002"}})
# CORS(app, resources={r"/PredictCropKPalay": {"origins": "http://localhost:5002"}})
# CORS(app, resources={r"/PredictCropTalong": {"origins": "http://localhost:5002"}})
# CORS(app, resources={r"/PredictCropPatatas": {"origins": "http://localhost:5002"}})

# define a route for the API


@app.after_request
def apply_caching(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,PUT,POST,DELETE,OPTIONS"
    return response

@app.route('/')
def render_index():
    # return "hello world";
    return render_template('index.html')


@app.route('/Agricrop', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])
    df['District_Name'] = district_loaded.transform(df['District_Name'])
    df['District_Name'] = df['District_Name'].astype(float)
    df['Crop'] = encoder.transform(df['Crop'])
    df['Crop'] = df['Crop'].astype(float)
    df['Area_Harvested'] = df['Area_Harvested'].astype(float)

    prediction = model.predict(df)[0]


    prediction = float(prediction)


    response = jsonify({'prediction': prediction})
    return response

@app.route('/PredictCrop', methods=['POST'])
# @cross_origin(origin='http://localhost:5002', headers=['Content-Type'])
def predictcrop():
    # Your prediction code here

    data = request.get_json()
    df = pd.DataFrame(data, index=[0])
    df['landType'] = landType_loaded.transform(df['landType'])
    df['landType'] = df['landType'].astype(float)
    df['seedType'] = seedType_loaded.transform(df['seedType'])
    df['seedType'] = df['seedType'].astype(float)
    df['harvestArea'] = df['harvestArea'].astype(float)

    # Here, you should use the relevant model for crop production prediction.
    # I see you have 'ampalayamodel', so you can use it like this:
    predictionP = ampalayamodel.predict(df)[0]


    predictionP = float(predictionP)


    response = jsonify({'predictionP': predictionP})
    return response

# Inside the /PredictCrop route
@app.route('/PredictCropKamatis', methods=['POST'])
# @cross_origin(origin='http://localhost:5002', headers=['Content-Type'])
def predictcrop_kamatis():
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])
    df['landType'] = landType_loaded.transform(df['landType'])
    df['landType'] = df['landType'].astype(float)
    df['seedType'] = seedType_loaded.transform(df['seedType'])
    df['seedType'] = df['seedType'].astype(float)
    df['harvestArea'] = df['harvestArea'].astype(float)


    predictionP = kamatismodel.predict(df)[0]


    predictionP = float(predictionP)


    response = jsonify({'predictionP': predictionP})
    return response

@app.route('/PredictCropCassava', methods=['POST'])
# @cross_origin(origin='http://localhost:5002', headers=['Content-Type'])
def predictcrop_cassava():
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])
    df['landType'] = landType_loaded.transform(df['landType'])
    df['landType'] = df['landType'].astype(float)
    df['seedType'] = seedType_loaded.transform(df['seedType'])
    df['seedType'] = df['seedType'].astype(float)
    df['harvestArea'] = df['harvestArea'].astype(float)

    predictionP = cassavamodel.predict(df)[0]

    predictionP = float(predictionP)

    response = jsonify({'predictionP': predictionP})
    return response

@app.route('/PredictCropMais', methods=['POST'])
# @cross_origin(origin='http://localhost:5002', headers=['Content-Type'])
def predictcrop_mais():
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])
    df['landType'] = landType_loaded.transform(df['landType'])
    df['landType'] = df['landType'].astype(float)
    df['seedType'] = seedType_loaded.transform(df['seedType'])
    df['seedType'] = df['seedType'].astype(float)
    df['harvestArea'] = df['harvestArea'].astype(float)


    predictionP = maismodel.predict(df)[0]


    predictionP = float(predictionP)


    response = jsonify({'predictionP': predictionP})
    return response

@app.route('/PredictCropPalay', methods=['POST'])
# @cross_origin(origin='http://localhost:5002', headers=['Content-Type'])
def predictcrop_palay():
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])
    df['landType'] = landType_loaded.transform(df['landType'])
    df['landType'] = df['landType'].astype(float)
    df['seedType'] = seedType_loaded.transform(df['seedType'])
    df['seedType'] = df['seedType'].astype(float)
    df['harvestArea'] = df['harvestArea'].astype(float)


    predictionP = palaymodel.predict(df)[0]


    predictionP = float(predictionP)


    response = jsonify({'predictionP': predictionP})
    return response

@app.route('/PredictCropTalong', methods=['POST'])
# @cross_origin(origin='http://localhost:5002', headers=['Content-Type'])
def predictcrop_talong():
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])
    df['landType'] = landType_loaded.transform(df['landType'])
    df['landType'] = df['landType'].astype(float)
    df['seedType'] = seedType_loaded.transform(df['seedType'])
    df['seedType'] = df['seedType'].astype(float)
    df['harvestArea'] = df['harvestArea'].astype(float)


    predictionP = talongmodel.predict(df)[0]


    predictionP = float(predictionP)


    response = jsonify({'predictionP': predictionP})
    return response

@app.route('/PredictCropPatatas', methods=['POST'])
# @cross_origin(origin='http://localhost:5002', headers=['Content-Type'])
def predictcrop_patatas():
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])
    df['landType'] = landType_loaded.transform(df['landType'])
    df['landType'] = df['landType'].astype(float)
    df['seedType'] = seedType_loaded.transform(df['seedType'])
    df['seedType'] = df['seedType'].astype(float)
    df['harvestArea'] = df['harvestArea'].astype(float)


    predictionP = patatasmodel.predict(df)[0]


    predictionP = float(predictionP)


    response = jsonify({'predictionP': predictionP})
    return response

if __name__ == '__main__':
    app.run(port = 5001)

