Train_chatbot.py — Dans ce fichier, nous allons construire et entraîner le modèle de Deep Learning qui peut classer et identifier ce que l’utilisateur demande au bot.

Gui_Chatbot.py — Ce fichier est l’endroit où nous allons construire une interface utilisateur graphique pour discuter avec notre chatbot formé.

Intents.json — Le fichier d’intentions contient toutes les données que nous utiliserons pour entraîner le modèle. Il contient une collection de balises avec leurs modèles et réponses correspondants.

Chatbot_model.h5 — Il s’agit d’un fichier de format de données hiérarchique dans lequel nous avons stocké les poids et l’architecture de notre modèle entraîné.

Classes.pkl — Le fichier pickle peut être utilisé pour stocker tous les noms de balises à classer lorsque nous prédisons le message.

Words.pkl — Le fichier de cornichon words.pkl contient tous les mots uniques qui constituent le vocabulaire de notre modèle.

https://dzone.com/articles/python-chatbot-project-build-your-first-python-pro





------------------------------------------ETAPES---------------------------------------------------
D'abord dans train_chatbot.py, nous allons separé chaque phrase de notre pattern en mot pour puis les stocker dans words et les balises dans classes.
Nous allons creer des données d entrenement et de sortie pour notre modele.
