# "Writing Custom Action" by RASADocs: https://rasa.com/docs/rasa/custom-actions
# "Conversational AI with Rasa: Custom Actions" by Rasa: https://www.youtube.com/watch?v=VcbfcsjBBIg
# "converting written date to date format in python" by Maratin Evans: https://stackoverflow.com/questions/43426021/converting-written-date-to-date-format-in-python
# "Get current date/time and compare with other date" by Kevin: https://stackoverflow.com/questions/32483997/get-current-date-time-and-compare-with-other-date
# "Cómo calcular el número de días entre dos fechas en Python" by Python: https://labex.io/es/tutorials/python-how-to-calculate-the-number-of-days-between-two-dates-in-python-395038

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import FollowupAction, SlotSet

from transformers import pipeline
from huggingface_hub import login
import torch
import requests
from http import HTTPStatus
import json
from datetime import datetime
import re

from pymongo import MongoClient

import subprocess
import webbrowser
import os

class ActionConversation(Action):
    
    pipe = None
    pipe_translation = None
    prompt_citas = \
    """
    Eres un asistente para personas que viven solas. Tus respuestas serán reproducidas por un altavoz, así que deben ser breves, claras y en español.

    El usuario te enviará un JSON con citas. Tu tarea es leerlo y explicar las citas de forma comprensible, en español completamente. No menciones que 
    se trata de un JSON ni repitas la estructura del mensaje.

    Ejemplo:
    Usuario: {{"FAC": ["Hospital de la Paz"], "DATE": ["7 de abril de 2025"], "TIME": ["12:00 PM"]}}
    Asistente: Tienes una cita el 7 de abril de 2025 en el Hospital de la Paz a las 12 PM.

    Ahora responde a la siguiente información:
    Usuario: %s
    Asistente:
    """
    prompt_citas_recordatorio = \
    """
    Eres un asistente personal para personas que viven solas. Tus respuestas deben ser breves, claras y en español, ya que se reproducirán por un altavoz. El objetivo es recordar al usuario las citas que tiene próximamente, usando un tono claro y directo.

    El usuario te enviará información sobre sus citas, y tu tarea es leer esa información y recordarle de manera comprensible, sin mencionar ni la estructura ni el formato de los datos.

    Ejemplo:
    Usuario: {{"FAC": ["Hospital de la Paz"], "DATE": ["7 de abril de 2025"], "TIME": ["12:00 PM"]}}
    Asistente: Recuerda que el 7 de abril de 2025 tienes una cita en el Hospital de la Paz a las 12 PM.

    Recuerda: Debes ofrecer la información de las citas como un recordatorio cercano en el tiempo, siempre en forma de **recordatorio** y **no como un aviso futuro distante**.

    Ahora, con la siguiente información, responde como un recordatorio de las citas:
    Usuario: %s
    Asistente:
    """
    prompt_asistente = \
    """
    Eres un asistente para personas que viven solas. Tus respuestas deben ser breves y claras, en español.

    Ejemplo:
    Usuario: ¿Dónde está París?
    Asistente: París está en Francia, y es su capital.

    Ahora responde a la siguiente pregunta:
    Usuario: %s
    Asistente:
    """
    
    prompt_entretenimiento = \
    """
    Actúa como un asistente personal de entretenimiento. Tu tarea es sugerir actividades de manera breve, cercana y estrictamente limitada a una lista disponible. Solo puedes recomendar actividades que estén relacionadas directamente con los gustos expresados por el usuario y que se encuentren en la lista de opciones disponibles. Si el usuario menciona cosas que **no le gustan**, no debes recomendar ninguna actividad relacionada con esos elementos. No asumas gustos que el usuario no haya mencionado explícitamente. No relaciones gustos sin sentido (por ejemplo, no asocies perros con jardinería o cocina). Si no encuentras gustos relacionados, debes resumir todas las opciones disponibles de forma clara y ordenada, sin inventar. No debes saludar, hacer preguntas abiertas ni agregar despedidas. Mantén un tono cercano y natural, como si continuaras una conversación, y utiliza frases cortas y claras. Las opciones de entretenimiento disponibles son: ejercicios mentales (Gbrainy); lectura (eBiblio); ejercicio físico (ejercicio general con vídeos en YouTube de Patry Jordan, y yoga con vídeos en YouTube de Xuan Lan); jardinería (web de InfoJardín); cocina (web de PetitChef); juegos, acertijos o adivinanzas (elhuevodechocolate.com); y escuchar música o Spotify (playlist animada).

    Ejemplo 1: Gustos: lectura, música. No le gusta: deportes. Mensaje del usuario: "¿Qué tienes de entretenimiento?" Respuesta: "Como te gusta la lectura, te recomiendo
    eBiblio, donde puedes encontrar libros, revistas y audiolibros."

    Ejemplo 2: Gustos: plantas, jardinería. No le gusta: ejercicio físico. Mensaje del usuario: "¿Qué actividades de entretenimiento ofrece?" Respuesta: "Te podría 
    interesar la web de InfoJardín para ver consejos sobre el cuidado de las plantas."

    Ejemplo 3: Gustos: perros, series de televisión. No le gusta: juegos de mesa. Mensaje del usuario: "¿Qué opciones de entretenimiento tienes?" Respuesta: 
    "No encontramos actividades directamente relacionadas con tus gustos. Puedes explorar: GBrainy para ejercicios mentales; eBiblio para lectura; vídeos de Patry Jordan para hacer ejercicio, o de Xuan Lan si estás interesado en el Yoga; La web de InfoJardín para jardinería; La web de PetitChef para cocina; juegos acertijos y adivinanzas en elhuevodechocolate.com; o escuchar música, disponemos de una playlist alegre en Spotify."

    Ejemplo 4 Gustos: chocolate, repostería. No le gusta: comida picante. Mensaje del usuario: "¿Qué opciones de entretenimiento puedo explorar?" Respuesta: "Como te 
    gusta la repostería y el chocolate, te recomiendo explorar recetas en la web de PetitChef, donde puedes encontrar ideas deliciosas para cocinar."

    Ahora, con base en la siguiente información, genera una respuesta siguiendo todas las instrucciones anteriores.

    Gustos del usuario:
    %s
    No le gusta:
    %s

    Mensaje reciente del usuario:
    %s
    """


    def __init__(self):
        super().__init__()
        #model_id = "meta-llama/Llama-3.2-1B-Instruct" 
        #login(token="TU_HUGGINGFACE_TOKEN_AQUI")
        #ActionConversation.pipe = pipeline(model=model_id, torch_dtype=torch.bfloat16, device_map="auto")
        ActionConversation.pipe_translation = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
        

    def name(self) -> Text:
        return "action_conversation"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        intent = tracker.latest_message.get ("intent")["name"]
        previous_intent = tracker.get_slot ("previous_intent")
        message = tracker.latest_message.get ("text")
        return ActionConversation.state_machine (message, intent, previous_intent, dispatcher,tracker)
        
    
    @staticmethod
    def state_machine (message, intent, previous_intent, dispatcher, tracker):
        
        first_interaction = tracker.get_slot ("first_interaction")
        if first_interaction:
            likes, calendar, dislikes, previous_conversation_intent = ActionConversation.obtener_memoria(tracker)
            lista_memoria = [SlotSet ("likes", likes), SlotSet ("calendar", calendar), SlotSet ("dislikes", dislikes), SlotSet ("previous_conversation_intent", previous_conversation_intent), SlotSet ("first_interaction", False)]
        else:
            lista_memoria = []
        
        if (intent == "anger" or intent == "angry_face"):
            return [SlotSet ("has_expressed_anger", True), SlotSet ("previous_intent", intent)] + lista_memoria
        elif (intent == "disgust" or intent == "disgust_face"):
            return [SlotSet ("has_expressed_disgust", True), SlotSet ("previous_intent", intent)]+ lista_memoria
        elif (intent == "fear" or intent == "fear_face"):
            return [SlotSet ("has_expressed_fear", True), SlotSet ("previous_intent", intent)]+ lista_memoria
        elif (intent == "joy" or intent == "happy_face"):
            return [SlotSet ("has_expressed_hapiness", True), SlotSet ("previous_intent", intent)]+ lista_memoria
        elif (intent == "sadness" or intent == "sad_face"):
            return [SlotSet ("has_expressed_sadness", True), SlotSet ("previous_intent", intent)]+ lista_memoria
        elif (intent == "surprise" or intent == "surprise_face"):
            return [SlotSet ("has_expressed_surprise", True), SlotSet ("previous_intent", intent)]+ lista_memoria
        elif (intent == "neutral"):
            return ActionConversation.neutral_state_machine (message, previous_intent, dispatcher, tracker)+ lista_memoria
        else:
            #response = ActionConversation.call_llm (prompt = ActionConversation.prompt_asistente % message)
            #dispatcher.utter_message(text = response)
            response = subprocess.run (["ollama", "run", "tinyllama"], input = ActionConversation.prompt_asistente % message, shell = True, capture_output = True, text = True, encoding = "utf-8")
            dispatcher.utter_message(text = response.stdout)
            return [SlotSet ("previous_intent", intent)]+ lista_memoria
        
    @staticmethod
    def obtener_memoria (tracker):
        try:
            client = MongoClient("mongodb+srv://test_user:test_user@rasa.rhkql5s.mongodb.net/?retryWrites=true&w=majority&appName=RASA")
            db = client["rasa_db"]
            collection = db["user_data"]
            #sender_id = tracker.sender_id
            sender_id = "test_user"
            
            user_info = collection.find_one({"sender_id": sender_id})

            if user_info:
                likes = user_info.get("likes", [])
                calendar = user_info.get("calendar", [])
                dislikes = user_info.get("dislikes", [])
                previous_conversation_intent = user_info.get ("previous_conversation_intent", None)
                return likes, calendar, dislikes, previous_conversation_intent
            else:
                return [], [], [], None
        
        except Exception as e:
            print ("Ha habido un error: " + e)
    
    @staticmethod
    def neutral_state_machine (message, previous_intent, dispatcher, tracker):
        greet_words = ["hola", "buenas", "buenos días", "buenas tardes", "buenas noches", "ey", "hey"]
        farewell_words = ["adiós", "chao", "chau", "hasta luego", "hasta pronto", "hasta la vista", "hasta mañana"]
        
        previous_conversation_intent = tracker.get_slot ("previous_conversation_intent")
        
        if any (greet_word in message.lower () for greet_word in greet_words):
            if (previous_conversation_intent in [
            "anger", "disgust", "fear", "sadness",
            "angry_face", "disgust_face", "fear_face", "sad_face",
            "dislike-anger", "dislike-disgust", "dislike-fear", "dislike-sadness"
            ]):
                dispatcher.utter_message (text = "Hola, ¿estás mejor?")
            else:
                dispatcher.utter_message (text = "Hola, ¿qué tal estas?")
                
            return []
        elif "ejercicios mentales" in message.lower (): #Ahora ejecuta la app, ya no menciona las otras opciones porque no eran viables
            dispatcher.utter_message (text = "Perfecto, voy a abrirte gbrainy y así podrás probar ejercicios de lógica, cálculo, memoria y acertijos verbales")
            ActionConversation.launch_gbrainy(message, dispatcher)  #añadido ejecucion apps
            return []
        elif "lectura" in message.lower ():
            dispatcher.utter_message (text = "Buena idea, te he abierto el catálogo de eBiblio Madrid para que accedas a los miles de audiolibros, libros electtrónicos y demás material que tienen.")
            ActionConversation.open_web_resource("https://madrid.ebiblio.es/resources")
            return []
        elif "ejercicio físico" in message.lower ():
            dispatcher.utter_message (text = "¡Genial! ¿Qué prefieres hacer, algo más de ejercicio general para fortalecerte o algo más relajado como yoga?")
            return []
        elif "ejercicio general" in message.lower() or "deporte" in message.lower():
            dispatcher.utter_message(text="¡Perfecto! Abriendo la lista de vídeos de rutina de brazos con mancuernas de Patry Jordan...")
            ActionConversation.open_web_resource("https://www.youtube.com/watch?v=ZIm_qrJSOds&list=PLReDUkUQG7xpCcu6kgK4RlppZkS-IV5EH&ab_channel=gymvirtual")
            return []
        elif "yoga" in message.lower():
            dispatcher.utter_message(text="¡Muy bien! Abriendo la lista de vídeos de clases de 15-30 minutos del canal de yoga de Xuan Lan...")
            ActionConversation.open_web_resource("https://www.youtube.com/watch?v=vzJ5gQrcag4&list=PL8n45gbMMrgU2cuCJJDVd_SOFtXf5x8xq&ab_channel=XuanLanYoga")
            return []
        elif "jardinería" in message.lower ():
            dispatcher.utter_message (text = "Muy bien, te estoy abriendo la web de infojardin para que puedas buscar las plantas que tienes y ver los mejores consejos de como cuidar de ellas.")
            ActionConversation.open_web_resource("https://articulos.infojardin.com/plantas/plantas.htm")
            return []
        elif "cocina" in message.lower ():
            dispatcher.utter_message (text = "Perfecto, te abro la web de PetitChef donde puedes buscar cientos de recetas subidas por sus usuarios." \
            "Puedes filtrar en base al tipo de comida: postre, entrante, plato, bebida..., buscar las mejor valoradas o incluso subir tu propia receta." \
            "Espero que encuentres algo que te guste.")
            ActionConversation.open_web_resource("https://www.petitchef.es/")
            return []
        elif "juegos" in message.lower() or "acertijos" in message.lower() or "adivinanzas" in message.lower():
            dispatcher.utter_message (text = "¡Buena elección! Te he abierto la página web de elhuevodechocolate, ahí tienes acertijos, adivinanzas y juegos centrados principalmente en niños pero que pueden entretener a cualquiera.")
            ActionConversation.open_web_resource("https://www.elhuevodechocolate.com/")
            return []
        elif "escuchar música" in message.lower () or "spotify" in message.lower(): #nueva funcionalidad añadida, llama a una playlist de spoti que tiene que estar instalada (la app)
            dispatcher.utter_message(text="¡Claro! Te voy a abrir una playlist de Spotify que seguro que te levanta el ánimo.")
            ActionConversation.launch_spotify_playlist(dispatcher)  # Llama a la función
            return []
        elif "entretenimiento" in message.lower ():
            gustos = tracker.get_slot ("likes")
            no_gustos = tracker.get_slot ("dislikes")
            if any ([gustos, no_gustos]):
                if gustos:
                    gustos_string = ", ".join(gustos)
                else:
                    gustos_string = "(sin especificar)"
                
                if no_gustos:
                    no_gustos_string = ", ".join(no_gustos)
                else:
                    no_gustos_string = "(sin especificar)"
                #response = ActionConversation.call_llm (prompt = ActionConversation.prompt_entretenimiento % (gustos_string, no_gustos_string, message))
                #dispatcher.utter_message(text = response)
                response = subprocess.run (["ollama", "run", "tinyllama"], input = ActionConversation.prompt_entretenimiento % (gustos_string, no_gustos_string, message), shell = True, capture_output = True, text = True, encoding = "utf-8")
                dispatcher.utter_message(text = response.stdout)
                return []
            else:
                dispatcher.utter_message(text = "Dispongo de las siguientes actividades: ejercicios mentales, lectura, ejercicio físico con Patry Jordan y Yoga con Xuan Lan, jardinería, cocina, juegos, acertijos, adivinanzas y música. Nombra alguna cuándo quieras hacerlas")
                return []
        elif "añadir un evento al calendario" in message.lower () or "añadir otro evento al calendario" in message.lower():
            return ActionConversation.extract_appointments(message, dispatcher, tracker)
        elif "citas" in message.lower ():
            info = tracker.get_slot ("calendar")
            info = json.dumps (info)
            #response = ActionConversation.call_llm (prompt = ActionConversation.prompt_citas % info)
            #dispatcher.utter_message(text = response)
            response = subprocess.run (["ollama", "run", "tinyllama"], input = ActionConversation.prompt_citas % info, shell = True, capture_output = True, text = True, encoding = "utf-8")
            dispatcher.utter_message(text = response.stdout)
            return []
        elif any (farewell_word in message.lower () for farewell_word in farewell_words):
            return ActionConversation.farewell_state_machine(previous_intent, dispatcher, tracker)
        else:
            #response = ActionConversation.call_llm (prompt = ActionConversation.prompt_asistente % message)
            #dispatcher.utter_message(text = response)
            response = subprocess.run (["ollama", "run", "tinyllama"], input = ActionConversation.prompt_asistente % message, shell = True, capture_output = True, text = True, encoding = "utf-8")
            dispatcher.utter_message(text = response.stdout)
            return []
    
    @staticmethod
    def farewell_state_machine (previous_intent, dispatcher, tracker):
        if (previous_intent in [
        "anger", "disgust", "fear", "sadness",
        "angry_face", "disgust_face", "fear_face", "sad_face",
        "dislike-anger", "dislike-disgust", "dislike-fear", "dislike-sadness"
        ]):
            dispatcher.utter_message (text = "Adiós. Espero que consigas mejorar tu estado de ánimo.")
        elif (previous_intent in [
        "joy", "happy_face", "like-joy"
        ]):
            dispatcher.utter_message (text = "Adiós. Me alegro de que hoy estés bien, ¡sigue así!.")
        else:
            dispatcher.utter_message (text = "Adiós.")
            
        calendario, near_events = ActionConversation.see_near_appointments (tracker)
        if (near_events):
            info = json.dumps (near_events)
            #response = ActionConversation.call_llm (prompt = ActionConversation.prompt_citas_recordatorio % info)
            #dispatcher.utter_message(text = response)
            response = subprocess.run (["ollama", "run", "tinyllama"], input = ActionConversation.prompt_citas_recordatorio % info, shell = True, capture_output = True, text = True, encoding = "utf-8")
            dispatcher.utter_message(text = response.stdout)
            
        
        ActionConversation.guardar_memoria(tracker)

        return calendario
        
        
    @staticmethod
    def extract_appointments (message, dispatcher, tracker):
        url = "http://localhost:50006/stanza-ner"
        try:
            msg = ActionConversation.translate (msg = message)
            response = requests.post (url= url, data= msg)
            if (response.status_code == HTTPStatus.OK ):
                response_json = json.loads (response.text) 
                if response_json:
                    if response_json.get ('DATE') is not None: 
                        date = ActionConversation.format_date (response_json.get ('DATE')[0])
                        if response_json.get ('TIME') is None:
                            time = "00:00"
                        else:
                            time = ActionConversation.format_time (response_json.get ('TIME')[0])
                        datetime_str = f"{date} {time}"
                        combined_datetime = datetime.strptime(datetime_str, "%d/%m/%Y %H:%M")
                        print (combined_datetime)
                        validated = ActionConversation.validate_date(combined_datetime)
                        if validated:
                            slot = tracker.get_slot ("calendar")
                            if slot is not None:
                                slot.append (response_json)
                                dispatcher.utter_message(text = "He añadido el evento al calendario, ¿algo más?")                                    
                                return [SlotSet("calendar", slot)]
                            else:
                                current_slot = list()
                                current_slot.append(response_json)
                                dispatcher.utter_message(text = "He añadido el evento al calendario, ¿algo más?")                                    
                                return [SlotSet("calendar", current_slot)]  
                        else:
                            dispatcher.utter_message(text = "No he añadido el evento al calendario porque la fecha está en el pasado")
                            return []
                    else: 
                        dispatcher.utter_message(text = "No he podido añadir el evento al calendario, ¿puedes repetirlo?")
                        return []
                else:
                    dispatcher.utter_message(text = "No he podido añadir el evento al calendario, ¿puedes repetirlo?") 
                    return []       
            else:
                dispatcher.utter_message(text = "No he podido añadir el evento al calendario, ¿puedes repetirlo?")
                return []
        except Exception:
            dispatcher.utter_message(text = "No he podido añadir el evento al calendario, ¿puedes repetirlo?")
            return []
            
           
    def validate_date (date):
        validated = False
        currentDate = datetime.now()
        if currentDate <= date:
            validated = True
        return validated
    
    def see_near_appointments (tracker):  
        events = tracker.get_slot ("calendar")
        currentDate = datetime.now()

        near_events = []
        valid_events = []
        if events is None:
            return [], None
        for event in events:
            date = event.get('DATE')
            time = event.get ('TIME')
            if date is not None:
                date = ActionConversation.format_date (str(date[0]))
                if time is None:
                    time = "00:00"
                else:
                    time = ActionConversation.format_time (str (time[0]))
                datetime_str = f"{date} {time}"
                combined_datetime = datetime.strptime(datetime_str, "%d/%m/%Y %H:%M")
                if currentDate <= combined_datetime:
                    valid_events.append (event)
                    time_left = combined_datetime - currentDate
                    days_left = time_left.days 
                    if days_left < 5:
                        near_events.append(event)

        return [SlotSet("calendar", valid_events)], near_events
                
    @staticmethod
    def format_date (date):
        formats = [
            "%B %d %Y",        # July 15 2024
            "%d %B %Y",        # 15 July 2024
            "%b %d %Y",        # Jul 15 2024
            "%m/%d/%Y",        # 07/15/2024
            "%m %d %Y",        # 07 15 2024
            "%B %d, %Y",       # July 15, 2024
            "%d %B, %Y",       # 15 July, 2024
            "%b %d, %Y",       # Jul 15, 2024
            "%m %d, %Y",       # 07 15, 2024
            "%A, %B %d %Y",    # Saturday, August 3 2024
            "%A, %B %d, %Y",   # Saturday, August 3, 2024
            "%A, %b %d %Y",    # Sat, Aug 3 2024
            "%A, %b %d, %Y",   # Sat, Aug 3, 2024
            "%A %B %d %Y",     # Saturday August 3 2024
            "%A, %B %d, %Y",    # Saturday August 3, 2024
        ]
        
        date = re.sub(r'(\d{1,2})(st|nd|rd|th)', r'\1', date.lower())
        
        print (date)
        
        year_pattern = r'\b\d{4}\b' 
        
        if not re.search(year_pattern, date):
            # Si no contiene un año, añadimos el año actual
            current_year = datetime.now().year
            date += f", {current_year}"
       
        print (date)
        
        for format in formats:
            try:
                date = datetime.strptime(date, format).strftime("%d/%m/%Y")
                print (date)
                return date
            except ValueError:
                continue
            

    @staticmethod
    def format_time(time):
        formats = ["%I:%M %p", "%I%p"]
        output_format = "%H:%M"
        time = time.upper ()
        for format in formats:
            try:
                time = datetime.strptime(time, format).strftime(output_format)
                print (time)
                return time
            except ValueError:
                continue
        return "00:00"

    @staticmethod
    def call_llm (prompt):
        try:
            outputs = ActionConversation.pipe(prompt, max_new_tokens=400, do_sample=True, top_k=10)

            generated_text = outputs[0]["generated_text"]

            if "Asistente:" in generated_text:
                respuesta = generated_text.split("Asistente:")[-1].strip()
            elif "Assistant:" in generated_text:
                respuesta = generated_text.split("Assistant:")[-1].strip()
            else:
                respuesta = generated_text.strip()

            print(respuesta)
            return respuesta
        except Exception:
            return "Lo siento, no te he entendido, puedes repetir."
        
    @staticmethod    
    def translate (msg):
        outputs = ActionConversation.pipe_translation(msg, max_new_tokens=400, do_sample=True, top_k=10)
        print (outputs[0]['translation_text'])
        return outputs[0]['translation_text']
        
    @staticmethod
    def guardar_memoria (tracker):
        try:
            client = MongoClient("mongodb+srv://test_user:test_user@rasa.rhkql5s.mongodb.net/?retryWrites=true&w=majority&appName=RASA")
            db = client["rasa_db"]
            collection = db["user_data"]
            
            # Obtener el sender_id
            #sender_id = tracker.sender_id
            sender_id = "test_user"
            
            # Recoger los slots de Rasa
            likes = tracker.get_slot("likes")
            calendar = tracker.get_slot ("calendar")
            dislikes = tracker.get_slot("dislikes")
            previous_intent = tracker.get_slot ("previous_intent")

            # Crear un documento con la información
            user_info = {
                "sender_id": sender_id,
                "likes": likes,
                "calendar": calendar,
                "dislikes": dislikes,
                "previous_conversation_intent": previous_intent
            }

            # Insertar o actualizar los datos en la base de datos
            collection.update_one({"sender_id": sender_id}, {"$set": user_info}, upsert=True)
            return 
        except Exception as e:
            print ("Ha habido un error: " + e)
    
    #Ejecución de la app de ejercicios mentales gbrainy (necesita estar instalada)
    @staticmethod
    def launch_gbrainy(message, dispatcher):
        exe_path  = r"C:\Program Files (x86)\gbrainy\gbrainy.exe"
        exe_dir = os.path.dirname(exe_path)  # Extrae el directorio: C:\Program Files (x86)\gbrainy
        try:
            subprocess.Popen([exe_path], cwd=exe_dir)  # <-- ¡Aquí el truco!
            dispatcher.utter_message(text="Ya he abierto gbrainy correctamente para que puedas empezar tus ejercicios mentales.")
        except Exception as e:
            dispatcher.utter_message(text="Lo siento, no he podido abrir gbrainy. Asegúrate de que lo tenga instalado si quieres entrenar tu mente.")
    
    #Ejecución de una playlist de spotify
    @staticmethod
    def launch_spotify_playlist(dispatcher):
        try:
            playlist_id = "37i9dQZF1DXdPec7aLTmlC"  #Este id es de la playlist "Temazos alegres", se puede cambiar para poner el que se quiera
            subprocess.Popen(["start", f"spotify:playlist:{playlist_id}"], shell=True)
            dispatcher.utter_message(text="¡Ya he abierto la playlist de Temazos alegres! Prueba a darle al play para ver si te levanta el ánimo.")
        except Exception as e:
            dispatcher.utter_message(text=f"No pude abrir la playlist. Error: {e}")

    #Ejecución de una url de página web
    @staticmethod
    def open_web_resource(url):
        webbrowser.open(url)

class SavePreferencesToSlot(Action):
    def name(self) -> Text:
        return "action_store_likes_dislikes"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        metadata = tracker.latest_message.get ("metadata", {})
        print (metadata)
        like = metadata.get ('like')
        dislike = metadata.get ('dislike')
        if like:
            likes = tracker.get_slot ("likes")
            if likes is not None:
                likes.append (like)                                  
                return [SlotSet("likes", likes)]
            else:
                likes = list()
                likes.append(like)                               
                return [SlotSet("likes", likes)]
        elif dislike:
            dislikes = tracker.get_slot ("dislikes")
            if dislikes is not None:
                print (dislike)
                dislikes.append (dislike)                                  
                return [SlotSet("dislikes", dislikes)]
            else:
                dislikes = list()
                dislikes.append(dislike)                               
                return [SlotSet("dislikes", dislikes)]
            
        
    