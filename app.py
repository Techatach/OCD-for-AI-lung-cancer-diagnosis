# py -3 -m venv .venv
# .venv\Scripts\activate
# pip install Flask
# pip install tensorflow==2.12.0
# หรือ pip install tensorflow==2.15.0
# pip install keras
# pip install Pillow
## คำสั่งรันโมเดล
# flask run 
# หรือ python -m flask run

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
# model = load_model('keras_model.h5')
model = load_model('keras_model.h5')
#model = load_model('SkinCancerFlask/efficientnetb3-Skin Cancer-69.39.h5')
# model = load_model(r'D:\SP_PublicAd\SP_2_2566\BreastCancer\BreastCancerCode\BreastCancerFlask\keras_model.h5')
target_img = os.path.join(os.getcwd() , 'static/images')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/index')
def index():
    return render_template('index.html')

#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png', 'gif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
def read_image(filename):

    img = load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path)
            class_prediction=model.predict(img) 
            classes_x=np.argmax(class_prediction,axis=1)
            if classes_x == 0:
              cancer = "lung_aca"
            elif classes_x == 1:
              cancer = "lung_bnt"
            else:
              cancer = "lung_scc"
            return render_template('predict.html', cancer = cancer, prob=class_prediction, user_image = file_path)
        else:
            return "Unable to read the file. Please check file extension"
          
####### Webcam
@app.route("/webcam")
def webcam():
    return render_template("webcam.html")

if __name__ == '__main__':
    app.run(debug=True)
    app.run(debug=True,use_reloader=False, port=5000)