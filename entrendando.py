import cv2
import os
import numpy as np

dataPath = '/home/pi/Facial recognition lock and web streaming monitoring Raspberry pi 4/Data' #Cambia a la ruta donde hayas almacenado Data
peopleList = os.listdir(dataPath)
print('People list: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
	personPath = dataPath + '/' + nameDir
	print('Read the samples')

	for fileName in os.listdir(personPath):
		print('Faces: ', nameDir + '/' + fileName)
		labels.append(label)
		facesData.append(cv2.imread(personPath+'/'+fileName,0))

	label = label + 1


# Métod for the recognition
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Training facial recognition
print("Training...")
face_recognizer.train(facesData, np.array(labels))

# Save the model
face_recognizer.write('modeloLBPHFace.xml')
print("Modelo almacenado...")