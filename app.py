from flask import Flask, render_template, request
import pickle
import numpy as np
from keras.models import model_from_json
#from tensorflow.keras.models import load_model


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify():
    text = str(request.form['sentence'])

    # load the saved model and tokenizer
    json_file = open('models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("models/model.h5")
    #saved_model = load_model('models/Intent_Classification.h5')
    Tokenizer = pickle.load(open('models/tokenizer.pkl', 'rb'))

    tokens = Tokenizer.texts_to_sequences([text])
    prediction = loaded_model.predict(np.array(tokens))
    pred = np.argmax(prediction)
    classes = ['BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook']
    result = classes[pred]

    return render_template("result.html", output = "The intent of the user is {}".format(result))


if __name__ == "__main__":
    app.run(debug=True)

