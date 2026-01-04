import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd
import requests
import time
import json
import pyttsx3

# Source:
# https://github.com/nicknochnack/MediaPipePoseEstimation/blob/main/Media%20Pipe%20Pose%20Tutorial.ipynb
    
def tts_function (texto:str) -> None:
            
    """
    Function that converts text to audio
            
    Parameters:
        - texto (str): name of the file that contains the audio.
            
    Returns: 
        -None
    """
            
    engine = pyttsx3.init ()
    engine.setProperty('rate', 150)  # Velocidad de habla
    engine.setProperty('volume', 1.0)  # Volumen 
    engine.say(texto)
    engine.runAndWait()

    return

def load_model ():
    with open('body_language.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def infer (model):
    last_class = None
    stability_counter = 0
    threshold = 100
    rasa_endpoint_url = "http://localhost:50007/recepcion-lenguaje-no-verbal"
    
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    columns = pd.read_csv("coords.csv", nrows=1).columns[1:]
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                # Extract Pose landmarks
                pose_landmarks  = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose_landmarks]).flatten())
            
                # Make Detections
                X = pd.DataFrame([pose_row], columns=columns)
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]

                if body_language_class == last_class:
                    stability_counter += 1
                else:
                    stability_counter = 0
                    last_class = body_language_class

                if stability_counter >= threshold:
                    # Evitar más envíos hasta nueva señal
                    stability_counter = 0
                    
                    headers = {
                        'Content-Type': 'application/json'
                    }
                    
                    data = {
                        "emotion": body_language_class,
                        "confidence": round(body_language_prob[np.argmax(body_language_prob)],2)
                    }
                    
                    sender = "test_user_body"
                    payload = {
                        "sender" : sender,
                        "message" : json.dumps(data)
                    }
                    
                    payload = json.dumps (payload)
                    response = requests.post(rasa_endpoint_url, data=payload, headers=headers)
                    response.raise_for_status()
                  
                    tts_function (response.text)

                    # Puedes usar un timer o una señal externa para reactivar el envío
                    time.sleep(5)  # Espera fija de 5 segundos
                
                # Grab ear coords
                coords = tuple(np.multiply(
                                np.array(
                                    (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x, 
                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y))
                            , [640,480]).astype(int))
                
                cv2.rectangle(image, 
                            (coords[0], coords[1]+5), 
                            (coords[0]+len(body_language_class)*20, coords[1]-30), 
                            (245, 117, 16), -1)
                cv2.putText(image, body_language_class, coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Get status box
                cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                
                # Display Class
                cv2.putText(image, 'CLASS'
                            , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0]
                            , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display Probability
                cv2.putText(image, 'PROB'
                            , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                            , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            except requests.exceptions.RequestException as e:
                print("Failed to send data to RASA:", e)
            except Exception as e:
                print (e)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
       
if __name__ == "__main__":
    model = load_model ()
    infer (model)