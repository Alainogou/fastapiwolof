from transformers import AutoTokenizer, TFAutoModelForTokenClassification, pipeline
import re
from typing import Optional
from fastapi import FastAPI

app = FastAPI()

# Charger le modèle pré-entraîné et le pipeline NER
tokenizer = AutoTokenizer.from_pretrained("masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0")
model = TFAutoModelForTokenClassification.from_pretrained("masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)



@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/text/{text_id}")
def read_item(text_id: str, q: Optional[str] = None):

    word_starts = get_word_starts(text_id)
    entities = assign_entities_to_words(word_starts, nlp(text_id))

    # Formatez la sortie sous forme de liste de dictionnaires
    ner_results = [{"word": word, "entity": entity} for word, entity in entities.items()]
    prediction_text = []  # Pour stocker les entités regroupées
    current_group = {"text": "", "entity": None}  # Pour stocker temporairement les entités en cours de regroupement

    for entity in ner_results:

        if entity['entity'] == 'O':
            current_group = {"text": entity['word'], "entity": 'O'}
        else:
            if entity['entity'].startswith("B-"):
                current_group = {"text": entity['word'], "entity": entity['entity'][2:]}  # Enlever le préfixe "B-"
            elif entity['entity'] == "I-" + current_group['entity']:
                # Ajouter l'entité à la phrase en cours de regroupement
                current_group['text'] += " " + entity['word']

        if not element_existe(current_group, prediction_text):
            prediction_text.append(current_group)


    return {"predict": prediction_text, "q":q}


def get_word_starts(sentence):
    words = re.findall(r'\S+', sentence)
    word_starts = {}
    start = 0

    for word in words:
        word_starts[start] = word
        start += len(word) + 1  # Ajouter 1 pour l'espace

    return word_starts




def assign_entities_to_words(word_dict, entity_list):
    entities = {}

    for word_start, word in word_dict.items():
        entities[word] = "O"  # Par défaut, l'entité est 0

        for entity in entity_list:
            if entity['start'] == word_start:
                entities[word] = entity['entity']
                break

    return entities


def element_existe(element, tableau):
    return element in tableau
