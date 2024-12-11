import cv2
import mediapipe as mp
import pickle 
import numpy as np


model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']


capture = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles 




hands = mp_hands.Hands(static_image_mode =True, min_detection_confidence =0.3)


labels_dict = {0: 'Apple', 1: 'Bread', 2: 'Help'}
while True:

    data = []
    X_=[]
    Y_ =[]

    ret, frame = capture.read()
    H,W,_= frame.shape
    frame_rgb= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(  #not needed?
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()

            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                data.append(x)
                X_.append(x)
                data.append(y)
                Y_.append(y)
        x1 = int(min(X_) * W) - 10
        y1 = int(min(Y_) * H) - 10

        x2 = int(max(X_) * W) - 10
        y2 = int(max(Y_) * H) - 10

        prediction = model.predict([np.asarray(data)])

        predicted_char = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(25)


capture.release()
cv2.destroyAllWindows
   
