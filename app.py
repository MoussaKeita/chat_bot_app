from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import json
from sklearn.preprocessing import LabelEncoder
#import tensorflow.python.framework.dtypes

app = Flask(__name__)

#chargement du model
with open("intents.json") as f:
    data = json.load(f)

training_sentences = []
traning_labels = []
responses = []
labels = []

for intent in data["intents"]:  # parcours du data
    for pattern in intent['patterns']:  # selection des phrases
        training_sentences.append(pattern)  # ajout des phrases
        traning_labels.append(intent['tag'])  # ajouts des labels
    responses.append(intent['responses'])  # ajouts des reponses

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

@app.route('/')
def home():
	return render_template('index.html')


label = LabelEncoder()

traning_labels = label.fit_transform(traning_labels)

@app.route('/predict',methods = ["POST"])
def predict():
    # loading model
    model = load_model('model_chatbot.h5')
    max_len = 20
    trunc_type = 'post'
    vocab_size = 10000
    oov_token = "<OOV>"
    saisi_user = request.form['query']
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    #tokenizer.fit_on_texts(saisi_user)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    result = model.predict(pad_sequences(tokenizer.texts_to_sequences([saisi_user]), truncating=trunc_type, maxlen=max_len))
    category = label.inverse_transform([np.argmax(result)])  # inverse_transform on the scaler to get the original unscaled data back.
    for i in data["intents"]:  # on parcours la data
        if i["tag"] == category:  # puis on compare le tag(label) avec la prediction
            reponse = np.random.choice(i["responses"])  # pour enfin afficher la reponse corespondante

    return render_template("index.html", response_text="{}".format(reponse))



if __name__ == "__main__":
    app.run(debug=True)