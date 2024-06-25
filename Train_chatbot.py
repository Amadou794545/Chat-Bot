import nltk
nltk.download('punkt')
nltk.download('wordnet')
import numpy as np
from keras import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import SGD
from nltk.stem import WordNetLemmatizer



lemmatizer = WordNetLemmatizer()

import json

intents_file = open('intents.json').read()
intents = json.loads(intents_file)

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(documents)

training_list = []
# Liste vide pour les sorties, de la longueur du nombre de classes
output_empty = [0] * len(classes)

# Parcourir chaque document
for doc in documents:
    bag = []
    word_patterns = doc[0]
    # Lemmatisation des mots
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # Création du sac de mots
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # Création de la ligne de sortie correspondante
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training_list.append([bag, output_row])

    # Ajout du sac de mots et de la ligne de sortie à la liste d'entraînement
    training_list.append([bag, output_row])
    # Conversion de la liste en tableau NumPy
    training = np.array(training_list, dtype=object)
    # Séparation des données d'entraînement et des étiquettes
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])



    print("Training data is created")

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(train_y[0]), activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("Model is created")
