from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time 
import sys
import mediapipe as mp
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pickle
from statistics import mode


def mediapipe_det (image , model):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False 
    results = model.process(img)                                        #detecting using mediapipe         
    image.flags.writeable = True 
    img = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)


    return image, results
def draw_lms (image, results):                                           #shows the actual mapping from mp
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def extractLMs(results):
    Plandmarks = np.array([[res.x, res.y,res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(32*3)
    LHlandmarks = np.array([[res.x, res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    RHlandmarks = np.array([[res.x, res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten()  if results.right_hand_landmarks else np.zeros(21*3)
    Flandmarks = np.array([[res.x, res.y,res.z] for res in results.face_landmarks.landmark]).flatten()  if results.face_landmarks else np.zeros(468*3)
    return np.concatenate([Plandmarks,LHlandmarks,RHlandmarks,Flandmarks])


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

labels_dict = {0: 'Apple', 1: 'Orange'}
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

##################run simulation on coppelia sim#########################

# while camera is still open
#sign recognition code
cap = cv2.VideoCapture(0)  
sign =[]
frames = []

print('Collecting images for sign ')

done = False
holistic = mp_holistic.Holistic(min_detection_confidence = 0.3, min_tracking_confidence = 0.3)
while cap.isOpened():
    ret, frame = cap.read()
    image, results = mediapipe_det(frame, holistic)
    cv2.putText(frame, 'Ready? Press "S" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                cv2.LINE_AA)
    cv2.imshow('Sign render frame', frame)
    if cv2.waitKey(10) == ord('s'):
        counter = 0
        s=[]
        while counter < 30:
            ret, frame = cap.read()
            cv2.imshow('Sign render frame', frame)
            s.append(extractLMs(results))
            cv2.waitKey(20)

            counter += 1
        for i in range (30):
            sign.append((np.array(s)))

        nsamples, nx, ny = np.array(sign).shape
        d2_sign = np.array(sign).reshape((nsamples,nx*ny))   


        result = mode( model.predict((np.array(d2_sign))))
        break

    


        #res = model.predict(x_test) 
cap.release()
cv2.destroyAllWindows()



# run simulation of object chosen

# connect to simulation
result=0
if result == 0:
    funct = 'bowl_path'
    funct2 ='bowl_thread'
if result == 1:
    funct ='cup_path'
    funct2 ='cup_thread'

print(labels_dict[result])
print("program started ")

clientID = RemoteAPIClient()
sim = clientID.require('sim')

sim.startSimulation()

if clientID!=-1:
    print("Connected to the remote API server")
else:
    print("Not connected to the remote API server")
    sys.exit("Could not connect")

NAOhandle =sim.getObject('/NAO/script')
count = 0
while count != 1000:
    sim.callScriptFunction(funct,NAOhandle)
    count +=1
sim.stopSimulation()
