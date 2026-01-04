import stanza    
from flask import Flask, request
import json

# Code sources:
# "Crea tu Primer Servidor con Python - Flask" by Facundo Carballo: https://www.youtube.com/watch?v=A7K1Cmwt9KI&t=440s
# "flask.Request.get_data" by FlaskAPI: https://tedboy.github.io/flask/generated/generated/flask.Request.get_data.html#flask.Request.get_data


app = Flask ("AURA")
nlp_calendar = stanza.Pipeline('en', processors='tokenize,ner')
nlp_likes_dislikes = stanza.Pipeline('es')

def obtain_dependencies(doc, complemento, max_depth):
    if max_depth == 0:  # Detener si se alcanza el límite de profundidad
        return
    
    for sent in doc.sentences:
        for word in sent.words:
            if word not in complemento and any(complem.id == word.head for complem in complemento):
                complemento.append(word)
                obtain_dependencies(doc, complemento, max_depth - 1)

def obtener_CD (doc, root, complemento):
    for sent in doc.sentences:
        for word in sent.words:
            if root.id == word.head:
                if word.deprel in ['obj', 'xcomp', 'nsubj', 'csubj', 'advcl']:
                    complemento.append (word)
                    obtain_dependencies (doc, complemento, 100)

@app.route('/stanza-ner', methods = ['POST'])
def obtain_data ():
        
    """
    Function that processes the auido transcripted.
        
    Parameters:
        - text (str): audio transcripted that has to be processed
        
    Returns: 
        - str: the response to the audio processed
    """
    texto = request.get_data (as_text=True)

    # Procesar el texto con la pipeline
    doc = nlp_calendar(texto)

    # Iterar sobre las entidades encontradas en el documento
    entidades_dict = dict ()
    for ent in doc.entities:
        tipo = ent.type
        texto = ent.text
        if tipo not in entidades_dict:
            entidades_dict[tipo] = []
        entidades_dict[tipo].append(texto)
            
    respuesta = json.dumps (entidades_dict)

    return respuesta

                    
@app.route('/stanza-likes-dislikes', methods = ['POST'])
def obtain_likes_and_dislikes():
    verbs_of_interest = {
        "gustar", "encantar", "fascinar", "apasionar", "interesar", "atraer",
        "agradar", "disfrutar", "deleitar", "entusiasmar", "entusiar", "molar",
        "odiar", "detestar", "molestar", "disgustar", "fastidiar",
        "aburrir", "aburrer", "hartar", "repugnar", "asquear", "cansar", "reventar", "soportar",
        "soporto", "odio"
    }
    
    text = request.get_data (as_text=True)
        
    doc = nlp_likes_dislikes(text)

    root = None
    for sent in doc.sentences:
        for word in sent.words:
            if word.lemma in verbs_of_interest:
                root = word
                break
                    
    complemento = list()            
    if root is not None:  
        obtener_CD (doc, root, complemento)
                        
    complemento.sort (key = lambda x: x.id, reverse=False)

    respuesta = str()
    for c in complemento:
        respuesta += c.text + " " 
        
    return respuesta

app.run(port = 50006)