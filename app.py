import json
from urllib import response
import numpy as np
from flask import Flask, request, jsonify, render_template
from pickle import load
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

app = Flask(__name__)


# bot=pickle.load(open('bot.pkl','rb'))
model = load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    For rendering results on HTML GUI
    """
    requestData = json.loads(request.data.decode("utf-8"))
    encoded = tokenizer.encode(
        str(requestData["text"]) + tokenizer.eos_token,
        return_tensors="pt",
    )
    model_rp = model.generate(
        encoded, max_length=1000, pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(
        model_rp[:, encoded.shape[-1] :][0],
        skip_special_tokens=True,
    )
    return "".join(response)


app.run(debug=True)
