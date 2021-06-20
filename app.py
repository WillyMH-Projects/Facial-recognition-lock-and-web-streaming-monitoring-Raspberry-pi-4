import cv2, os, imutils, io, time
from flask import Flask, render_template, Response
import RPi.GPIO as GPIO
from RPLCD.gpio import CharLCD
import numpy as np

#LCD initialitation
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(10, GPIO.IN)
lcd = CharLCD(cols=16, rows=2, pin_rs=37, pin_e=35, pins_data=[40, 38, 36, 32],numbering_mode=GPIO.BOARD)
GPIO.setup(8, GPIO.OUT, initial=GPIO.LOW)

GPIO.output(8, GPIO.HIGH)

dataPath = '/home/pi/RFacial recognition lock and web streaming monitoring Raspberry pi 4/Data' #Cambia a la ruta donde hayas almacenado Data
imagePaths = os.listdir(dataPath)

# Read the model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPHFace.xml')
cap = cv2.VideoCapture(0)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

#Web streaming
app = Flask(__name__)
sub = cv2.createBackgroundSubtractorMOG2()  # create background subtractor

def encender():
	abrir()

def apagar():
	cerrar()

def abrir():
	lcd.clear()
	lcd.write_string(u'OPEN')
	rele()

def cerrar():
	lcd.clear()
	lcd.write_string(u'Unknown')
	norele()

def rele():
	GPIO.output(8, GPIO.LOW)

def norele():
	GPIO.output(8, GPIO.HIGH)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen():
	while True:
		ret,frame = cap.read()
		frame =  imutils.resize(frame, width=640)
		if GPIO.input(10)==GPIO.LOW:
			lcd.clear()
			lcd.write_string(u'Unknown')
			GPIO.output(8, GPIO.HIGH)
		if ret == False: 
			break
		else:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			auxFrame = gray.copy()
			faces = faceClassif.detectMultiScale(gray,1.3,5)

			for (x,y,w,h) in faces:
				rostro = auxFrame[y:y+h,x:x+w]
				rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
				result = face_recognizer.predict(rostro)
				cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
		
				# LBPHFace
				if result[1] < 70:
					cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
					cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
					encender()

				else:
					cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
					cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
					apagar()

			
			frame = cv2.imencode('.jpg', frame)[1].tobytes()
			yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
