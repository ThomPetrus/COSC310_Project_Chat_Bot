
#load packages

from rasa.nlu.training_data import load_data
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Trainer
from rasa.nlu import config
import spacy



# load data
print("Loading training data...")
train_data = load_data('rasa_dataset.json')

#config.load("config_spacy.yaml")

trainer = Trainer(config.load("config_spacy.yaml"))

## Training Data
#trainer.train(train_data)