from flask import Flask, request, jsonify
from threading import Condition 
import json
import requests
from http import HTTPStatus

# Code sources:
# "Crea tu Primer Servidor con Python - Flask" by Facundo Carballo: https://www.youtube.com/watch?v=A7K1Cmwt9KI&t=440s
# "flask.Request.get_data" by FlaskAPI: https://tedboy.github.io/flask/generated/generated/flask.Request.get_data.html#flask.Request.get_data
# https://www.youtube.com/watch?v=PhUrWAc5-30&t=176s

app = Flask("Integrador")

condicion = Condition()
voz_autoriza = {'activo': True}

PALABRA_CLAVE = "Adiós"

def procesar_peticiones(datos: dict):
    url = "http://localhost:50005/webhooks/rest/webhook"
    sender = "test_user"
    message = datos.get('message')

    payload = json.dumps({'sender': sender, 
                          'message': message})
        
    response = requests.post(url=url, data=payload)
    if (response.status_code == HTTPStatus.OK ):
        response_json = json.loads (response.text) 
        answer = ""
        for r in response_json: 
            answer += r['text'].encode("utf-8").decode () + ". " 
            
        return answer
    else:
        return "There has been an error."

@app.route('/recepcion-voz', methods=['POST'])
def recepcion_voz():
    datos = request.get_json()

    with condicion:
        voz_autoriza['activo'] = False
        respuesta = procesar_peticiones (datos)

        if PALABRA_CLAVE in respuesta:
            voz_autoriza['activo'] = True
            condicion.notify_all()
            
    return respuesta

@app.route('/recepcion-lenguaje-no-verbal', methods=['POST'])
def canal_restringido():
    datos = request.get_json()

    with condicion:
        if voz_autoriza['activo']:
            respuesta = procesar_peticiones (datos)
            return respuesta
        else:
            return jsonify({"error": "Canal bloqueado por voz."}), 403
            


app.run(port=50007)