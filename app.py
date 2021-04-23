from flask import Flask,render_template, request
import numpy as np
import cv2
from keras.models import load_model
model=load_model('SCKDMarch26th.h5')
facecascade=cv2.CascadeClassifier('haarcascade_frontalface.xml')
app = Flask(__name__)

@app.route('/', methods = ["GET","POST"])
def index():
    if request.method == "GET":
        return render_template('index.html')
    else:
        image = request.files["file"].read()
        image = np.fromstring(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        face=facecascade.detectMultiScale(image,1.1,5)
        # print(face)
        for feature in face:
            # print('worked')
            cv2.rectangle(image,
                        (feature[0],feature[1]),
                        (feature[0]+feature[2],
                        feature[1]+feature[3]),(0,255,0),2)
            ROI = image[feature[1]:feature[1]+feature[3],
                            feature[0]:feature[0]+feature[2]]
            ROI=cv2.resize(ROI,(50,50))
            dilatedimg=cv2.dilate(ROI,(3,3))
            dilatedlist=[dilatedimg] #shape needs to be array so made it a list
            dilatedarray=np.array(dilatedlist) #shape needs to be (1,50,50,3) so we make it an array
            # print(dilatedarray.shape)
            dilatedarray=dilatedarray/255 #range only from 0 to 1 so more accurate
            
            predictions=model.predict(dilatedarray)

            players=['Kevin Durant','Stephen Curry']
            # print(predictions)
            maximum=np.argmax(predictions[0])
            # print(players[maximum])

            cv2.putText(image, str(players[maximum]), (feature[0]-100,feature[1]+50),cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2) 
            
        if face != ():
            cv2.imwrite('static/image.png',image)
            return render_template('showPrediction.html')
        else:
            return "no face detected"





if __name__ == '__main__':
    app.run()