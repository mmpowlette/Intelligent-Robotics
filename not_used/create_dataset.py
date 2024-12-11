import os 
import mediapipe as mp 
import cv2
import pickle
import matplotlib.pyplot as plt 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles 


hands = mp_hands.Hands(static_image_mode =True, min_detection_confidence =0.3)


DATA_DIR = './data'
LMdata=[]
LMlabels=[]

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR,dir_)):
        LMdata_aux = []
        img = cv2.imread(os.path.join(DATA_DIR,dir_,img_path))
        img_rgb= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    X = hand_landmarks.landmark[i].x
                    Y= hand_landmarks.landmark[i].y
                    LMdata_aux.append(X)
                    LMdata_aux.append(Y)




        LMdata.append(LMdata_aux)
        LMlabels.append(dir_)

file = open('data.pickle','wb')
pickle.dump({'data': LMdata, 'labels': LMlabels},file)
file.close()