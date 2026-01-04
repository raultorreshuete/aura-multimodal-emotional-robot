from deepface import DeepFace
import cv2
import mediapipe as mp
import requests
import time
import json
import pyttsx3

# Fuente: 
#https://github.com/AprendeIngenia/Reconocimiento-de-Caracteristicas-Humanas/blob/main/Deep.py

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

def face_analisis():
    last_class = None
    stability_counter = 0
    threshold = 20
    rasa_endpoint_url = "http://localhost:50007/recepcion-lenguaje-no-verbal"
    
    mp_face = mp.solutions.face_detection

    cap = cv2.VideoCapture(0)
    with mp_face.FaceDetection(min_detection_confidence=0.8, model_selection=0) as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(image_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = x1 + int(bbox.width * w)
                    y2 = y1 + int(bbox.height * h)

                    # Dibuja el rectángulo en la cara
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Recorta la cara para analizarla con DeepFace
                    face_crop = frame[y1:y2, x1:x2]

                    try:
                        info = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False, detector_backend="centerface")
                        dominant_emotion = info[0].get('dominant_emotion')
                        emotion_confidence = info[0]['emotion'][dominant_emotion]
                        emotion = f"{dominant_emotion}, {round(emotion_confidence, 2)}"
                        
                        if dominant_emotion == last_class:
                            stability_counter += 1
                        else:
                            stability_counter = 0
                            last_class = dominant_emotion
                        
                        if stability_counter >= threshold:
                            # Evitar más envíos hasta nueva señal
                            stability_counter = 0
                            
                            headers = {
                                'Content-Type': 'application/json'
                            }
                            
                            data = {
                                "emotion": dominant_emotion,
                                "confidence": round(float (emotion_confidence)/100,2)
                            }
                            
                            sender = "test_user_face"
                            payload = {
                                "sender" : sender,
                                "message" : json.dumps(data)
                            }
                            
                            payload = json.dumps (payload)
                            response = requests.post(rasa_endpoint_url, data=payload, headers=headers)
                            response.raise_for_status()
                            tts_function (response.text)

                            # Puedes usar un timer o una señal externa para reactivar el envío
                            time.sleep(5)
                        
                        # Escribe la emoción al lado del rostro
                        cv2.putText(frame, emotion, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    except requests.exceptions.RequestException as e:
                        print("Failed to send data to RASA:", e)
                    except Exception as e:
                        print("Error en DeepFace:", e)
                        pass

                    # Solo una cara para reducir carga
                    break

            cv2.imshow("Detección de Emoción", frame)
            if cv2.waitKey(5) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_analisis()