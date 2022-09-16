from urllib import response
import numpy as np
from flask import Flask, request, jsonify, render_template
from pickle import load
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from module import ChatBot

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

app = Flask(__name__)


# bot=pickle.load(open('bot.pkl','rb'))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    For rendering results on HTML GUI
    """
    # user_input = tokenizer.encode(str(request.form.values())+ tokenizer.eos_token,return_tensors='pt')
    # prediction= model.generate(user_input,max_length=1000,pad_token_id=tokenizer.eos_token_id)
    # prediction2 = tokenizer.decode(prediction[:,user_input.shape[-1]:],skip_special_tokens=True)
    # input=bot.user_input(request.form.values())
    # response=bot.bot_response(input)
    print("Request : ", request.form)
    input_id = request.form.values()
    encoded = tokenizer.encode(str(input_id) + tokenizer.eos_token, return_tensors="pt")
    model_rp = model.generate(
        encoded, max_length=1000, pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(
        model_rp[
            :,
        ][0],
        skip_special_tokens=True,
    )
    input_id = ""

    return "DATA WILL BE HERE!!!"

    # return render_template('index.html', prediction_text='Response {}'.format(response))


app.run(debug=True)
model = load(open("model.pkl", "rb"))
