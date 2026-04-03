import cv2
import os
from flask import Flask,request,render_template,redirect
import numpy as np
from tensorflow.keras.models import load_model

app=Flask(__name__)
app.config['upload_folder']="uploads"
#loading the model
model = load_model('mri_model.h5')

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128,128))
    img = img / 255.0
    img = np.reshape(img, (1,128,128,3))

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        return "Tumor Detected"
    else:
        return "No Tumor"
    
 
@app.route('/',methods=['GET','POST'])
def default():
   return render_template("home.html")

@app.route('/predict',methods=['POST']) 
def predict():
   img= request.files['image']  
   path=os.path.join(app.config['upload_folder'],'upload.jpg')
    
   img.save(path)
   result=predict_image(path)
    
   if result is not None:
      return render_template("result.html",post=result)
   return redirect('/')  

if __name__=="__main__":
   app.run(debug=True)
