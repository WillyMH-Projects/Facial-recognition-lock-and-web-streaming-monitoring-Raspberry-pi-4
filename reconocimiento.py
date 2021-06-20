import cv2
import os
import imutils
import RPi.GPIO as GPIO
from RPLCD.gpio import CharLCD

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
lcd = CharLCD(cols=16, rows=2, pin_rs=37, pin_e=35, pins_data=[40, 38, 36, 32],numbering_mode=GPIO.BOARD)
GPIO.setup(8, GPIO.OUT, initial=GPIO.LOW)

dataPath = '/home/pi/Facial recognition lock and web streaming monitoring with Raspberry pi 4/Data' #Cambia a la ruta donde hayas almacenado Data
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Leyendo el modelo
face_recognizer.read('modeloLBPHFace.xml')

cap = cv2.VideoCapture(0)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
	#lcd.clear()
	#lcd.write_string(u'Cerrado')	
	ret,frame = cap.read()
	frame =  imutils.resize(frame, width=640)
	if ret == False: break
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
			a=1
		else:
			cv2.putText(frame,'Unknown',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
			a=0

		if a==1:
			lcd.clear()
			lcd.write_string(u'Open')


		else:
			lcd.clear()
			lcd.write_string(u'CLOSE')


		if a==1:
			GPIO.output(8, GPIO.LOW)

		else:
			GPIO.output(8, GPIO.HIGH)


	
	cv2.imshow('frame',frame)
	k = cv2.waitKey(1)
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()