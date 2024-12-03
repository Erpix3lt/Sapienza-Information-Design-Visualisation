# Information Design
Vulnerability in OpenSource projects, has been and will be a problem. Sufficient EU funding is needed, in order to accelerate OpenSource projects in the EU.
## Collecting Data
- **DATASET I**  [EU Open-Source](https://digital-strategy.ec.europa.eu/en/library/study-about-impact-open-source-software-and-hardware-technological-independence-competitiveness-and)
- **DATASET II**  [npm-follower](https://arxiv.org/pdf/2308.12545)

### Research Data Retrieval
In order to construct our first dataset, we are planning on using language models to extract data from the EU Open-Source research paper. As this is the most valueable data source we could find.
Previously I have used a similar approach to retrieve animal descriptions from books. This might be helpfull: https://github.com/Erpix3lt/KISD-Envisioning-Non-Human-Centered-Perspectives.

```python
import spacy
from TextLoader import TextLoader
from CharacterEntity import CharacterEntity
from Database import DatabaseHandler
from AnimalsToObserve import animals_to_observe
from sklearn.feature_extraction.text import TfidfVectorizer

class Analyser:

    def __init__(self, db_handler, plain_text_url, text_name):
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.max_length = 1500000  # Set a higher limit (adjust as needed)
        self.character_entities = []
        self.entities_to_observe = ['PERSON']
        self.animals_to_observe = animals_to_observe
        self.text_loader = TextLoader(plain_text_url)
        self.text = self.text_loader.load()
        self.table_name = text_name
        self.db_handler = db_handler

    def retrieve_entities(self, text=None):
        if text is None:
            text = self.text

        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in self.entities_to_observe:
                self.process_entity(ent)

    def process_entity(self, entity):
        descriptive_text = self.extract_descriptive_text(entity)
        is_found, animals_found = self.filter_animal_desciptions(descriptive_text)
        if is_found:
            character = CharacterEntity(entity.text, entity.label_, animals_found, [descriptive_text.split('&&')], 1)
            existing_entity = next((e for e in self.character_entities if e.name == character.name), None)
            if existing_entity:
                existing_entity.num_occurrences += 1
                existing_entity.descriptive_text.append(descriptive_text.split('&&'))
            else:
                self.character_entities.append(character)

    def extract_descriptive_text(self, entity):
        sentences = list(entity.doc.sents)
        sentence_index = next((i for i, sent in enumerate(sentences) if entity.start_char < sent.end_char), None)

        if sentence_index is not None:
            before_sentence = sentences[sentence_index - 1].text if sentence_index > 0 else ''
            entity_sentence = sentences[sentence_index].text
            after_sentence = sentences[sentence_index + 1].text if sentence_index + 1 < len(sentences) else ''

            descriptive_text = f"{before_sentence}&&{entity_sentence}&&{after_sentence}"
            return descriptive_text.strip()
        return ''

    def filter_animal_desciptions(self, descriptive_text):
        is_found = False
        animals_found = []
        if descriptive_text is None:
            return False, animals_found
        for animal in self.animals_to_observe:
            if f" {animal.lower()} " in f" {descriptive_text.lower()} ":
                is_found = True
                animals_found.append(animal)
        return is_found, animals_found

if __name__ == "__main__":
    db_handler = DatabaseHandler()
    plain_text_url = "https://archive.org/stream/bub_gb_Mx4W0HHMJAYC/bub_gb_Mx4W0HHMJAYC_djvu.txt"
    table_name = "the_fables_of_bidpai"
    analyser = Analyser(db_handler, plain_text_url, table_name)
    analyser.retrieve_entities()
    print(f"Found {len(analyser.character_entities)} entities")
    db_handler.insert_character_entities(table_name, analyser.character_entities)
    print("Analysis complete")
```
