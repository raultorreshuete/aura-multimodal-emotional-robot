from rasa.shared.nlu.training_data.message import Message
from typing import Any, List, Optional, Text, Dict
from rasa.engine.graph import GraphComponent
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.recipes.default_recipe import DefaultV1Recipe

from transformers import pipeline
import requests
import json

# "Custom Graph Components" by Maxime Vdb: https://legacy-docs-oss.rasa.com/docs/rasa/custom-graph-components/
# "An Introduction to Python Subprocess: Basics and Examples" by Moez Ali: https://www.datacamp.com/tutorial/python-subprocess?dc_referrer=https%3A%2F%2Fwww.google.com%2F&utm_source=google&utm_medium=paid_search&utm_campaign=230119_1-sea%7Edsa%7Etofu_2-b2c_3-es-lang-en_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na
# "Emotion English DistilRoBERTa-base": https://huggingface.co/j-hartmann/emotion-english-distilroberta-base

@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER], is_trainable=False
)
class IntentClassifier (GraphComponent):
    """A custom intent classifier analysis component"""
    name = "intent-classifier"
    provides = ["intents"]
    requires = []
    defaults = {}
    language_list = ["es"]
    
    @classmethod
    def create( cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, ) -> GraphComponent:
        return cls(model_storage, resource, training_artifact=None)
    
    def __init__( self, model_storage: ModelStorage, resource: Resource, training_artifact: Optional[Dict],) -> None:
        self._model_storage = model_storage
        self._resource = resource
        
        self.pipe_translation = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
        self.pipe_classification = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
        self.pipe_likes_dislikes = pipeline("zero-shot-classification", model="Recognai/zeroshot_selectra_medium")
        print ("initialised the class")
    
    def process(self, messages: List[Message]) -> List[Message]:

        try:
            for message in messages:
                if "{" in message.get ('text') or "}" in message.get ('text'):
                    emotion_data = json.loads (message.get ('text'))
                    winner_intent = {"name": emotion_data.get("emotion") + "_face", "confidence": emotion_data.get ("confidence")}
                    message.set (
                        "intent",
                        winner_intent,
                        add_to_output=True
                    )
                else:
                    response = self.translate (msg = message.get ('text'))
                    
                    intents = self.pipe_classification (response)[0]
                    winner_intent = {"name": None, "confidence": 0.0}
                        
                    for intent in intents:
                        if intent.get ("score") > winner_intent.get ("confidence"):
                            winner_intent["name"] = intent.get ("label")
                            winner_intent["confidence"] = intent.get ("score")
                    
                    result = self.pipe_likes_dislikes(
                        message.get ('text'),
                        candidate_labels=["agrado", "desagrado", "otro"],
                        hypothesis_template="Este texto expresa {}."
                    )
                    
                    url = "http://localhost:50006/stanza-likes-dislikes"
                    like = None
                    dislike = None
                    if result['labels'][0] == 'agrado' and result ['scores'][0] >= 0.5:
                        like = requests.post (url= url, data= message.get ('text')).text
                            
                    elif result['labels'][0] == 'desagrado' and result ['scores'][0] >= 0.5:
                        dislike = requests.post (url= url, data= message.get ('text')).text
                    
                    if like:
                        message.set (
                                "metadata",
                                {"like": like},
                                add_to_output=True
                            )
                            
                        message.set (
                                "intent",
                                {"name": "like-" + winner_intent['name'], "confidence": 1.0},
                                add_to_output=True
                            )
                    elif dislike:
                        message.set (
                                "metadata",
                                {"dislike": dislike},
                                add_to_output=True
                            )
                            
                        message.set (
                                "intent",
                                {"name": "dislike-" + winner_intent['name'], "confidence": 1.0},
                                add_to_output=True
                            )
                    else: 
                        message.set (
                            "intent",
                            winner_intent,
                            add_to_output=True
                        )
                        
        except Exception:
            message.set (
                    "intent",
                    {"name": "neutral", "confidence": 0.5},
                    add_to_output=True
                )
        finally: 
            return messages
    
    def translate (self, msg):

        outputs = self.pipe_translation(msg, max_new_tokens=400, do_sample=True, top_k=10)
            
        return outputs[0]['translation_text']
        