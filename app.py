from flask import Flask, render_template, request
import pickle
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")


@app.route("/classify", methods=['GET', 'POST'])
def classify():
    text = str(request.form['sentence'])

    # load the saved model and tokenizer
    model = keras.models.load_model('models/Intent_Classification.h5')
    tokenizer = pickle.load(open('models/tokenizer.pkl', 'rb'))

    tokens = tokenizer.texts_to_sequences([text])
    tokens = pad_sequences(tokens, maxlen=100)
    prediction = model.predict(np.array(tokens))
    pred = np.argmax(prediction)
    classes = ['BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook']
    result = classes[pred]

    return render_template("result.html", output="The intent of the user is {}".format(result))


if __name__ == "__main__":
    app.run(debug=True, port=5544)
