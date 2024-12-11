import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
import mediapipe as mp 
import pickle 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


signs = np.array(['apple', 'orange' ])
no_sequences = 30
sequence_length = 30


# make folders for everything and save all images for each sign

for sign in signs:
    for sequence in range(no_sequences):
        try: os.makedirs(os.path.join(DATA_DIR, sign, str(sequence)))
        except: pass

for sign in signs:
    for sequence in range (no_sequences):
            cap = cv2.VideoCapture(0) 
            print('Collecting images for sign {}'.format(sign))

            done = False
            model = mp_holistic.Holistic(min_detection_confidence = 0.3, min_tracking_confidence = 0.3)
            while cap.isOpened():
                ret, frame = cap.read()
                image, results = mediapipe_det(frame, model)
                draw_lms(image, results)
                cv2.putText(frame, 'Ready? Press "S" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                            cv2.LINE_AA)
                cv2.imshow('Sign render frame', frame)
                if cv2.waitKey(25) == ord('s'):
                    break

            counter = 0
            while counter < sequence_length:
                ret, frame = cap.read()
                print('Say {}'.format(sign))
                result_test = extractLMs(results)
                np.save(os.path.join(DATA_DIR, sign, str(sequence),'{}'.format(counter)),result_test)
                cv2.waitKey(50)
                counter +=1
                    
                



cap.release()
cv2.destroyAllWindows()


label_map = {label: num for num, label in enumerate (signs)}

sequenceA, labelsA = [], []
for sign in signs:
    for sequence in range(no_sequences):
        framesA = []
        
        for frame in range(sequence_length):
            res = np.load(os.path.join(DATA_DIR, sign, str(sequence),'{}.npy'.format(frame)))
            framesA.append(res)

        sequenceA.append(framesA)
        labelsA.append(label_map[sign])

nplabelsA= np.array(labelsA)
npsequenceA =np.array(sequenceA)
print(npsequenceA.shape)
nsamples, nx, ny = npsequenceA.shape
d2_sequenceA = npsequenceA.reshape((nsamples,nx*ny))      ###problem?

##  making the data base for test 

x_train, x_test, y_train, y_test = train_test_split(d2_sequenceA, nplabelsA, test_size=0.2, shuffle=True, stratify=nplabelsA)

model = RandomForestClassifier()
# print(y_train.shape)

model.fit(x_train, y_train)
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))


print(x_test.shape)
print(x_train.shape)


# f = open('model.p', 'wb')
# pickle.dump({'model': model}, f)
# f.close()
























# print(results.face_landmarks.landmark)

# draw_lms(frame , results)
# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# cv2.imshow('Sign render frame', frame)
