#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://huggingface.co/microsoft/DialoGPT-medium?text=Hey+my+name+is+Julien%21+How+are+you%3F
# get_ipython().system('pip install transformers')
# get_ipython().system('pip install torch')


# In[2]:


import numpy as np
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pickle5 as pickle


# In[3]:


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


# In[4]:


# Build a ChatBot class with all necessary modules to make a complete conversation
class ChatBot:
    # initialize
    def __init__(self):
        # once chat starts, the history will be stored for chat continuity
        self.chat_history_ids = None
        # make input ids global to use them anywhere within the object
        self.bot_input_ids = None
        # a flag to check whether to end the conversation
        self.end_chat = False

    def user_input(self):
        # receive input from user
        text = input("User    >> ")
        # end conversation if user wishes so
        if text.lower().strip() in ["bye", "quit", "exit"]:
            # turn flag on
            self.end_chat = True
            # a closing comment
            print("ChatBot >>  See you soon! Bye!")
            time.sleep(1)
            print("\nQuitting ChatBot ...")
        else:
            # continue chat, preprocess input text
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            self.new_user_input_ids = tokenizer.encode(
                text + tokenizer.eos_token, return_tensors="pt"
            )

    def bot_response(self):
        # append the new user input tokens to the chat history
        # if chat has already begun
        if self.chat_history_ids is not None:
            self.bot_input_ids = torch.cat(
                [self.chat_history_ids, self.new_user_input_ids], dim=-1
            )
        else:
            # if first entry, initialize bot_input_ids
            self.bot_input_ids = self.new_user_input_ids

        # define the new chat_history_ids based on the preceding chats
        # generated a response while limiting the total chat history to 1000 tokens,
        self.chat_history_ids = model.generate(
            self.bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id
        )
        # print(type(self.chat_history_ids),'hii')
        # last ouput tokens from bot
        response = tokenizer.decode(
            self.chat_history_ids[:, self.bot_input_ids.shape[-1] :][0],
            skip_special_tokens=True,
        )
        # in case, bot fails to answer
        # print(type(response),'23')
        # print(self.chat_history_ids[:, self.bot_input_ids.shape[-1]:],self.chat_history_ids.shape)
        # print(type(self.chat_history_ids[:, self.bot_input_ids.shape[-1]:][0]),'ok')
        if response == "":
            response = self.random_response()
        # print bot response
        # print('ChatBot >>  '+ response)
        return response

    # in case there is no response from model
    def random_response(self):
        i = -1
        response = tokenizer.decode(
            self.chat_history_ids[:, self.bot_input_ids.shape[i] :][0],
            skip_special_tokens=True,
        )
        # iterate over history backwards to find the last token
        while response == "":
            i = i - 1
            response = tokenizer.decode(
                self.chat_history_ids[:, self.bot_input_ids.shape[i] :][0],
                skip_special_tokens=True,
            )
        # if it is a question, answer suitably
        if response.strip() == "?":
            reply = np.random.choice(["I don't know", "I am not sure"])
        # not a question? answer suitably
        else:
            reply = np.random.choice(["Great", "Fine. What's up?", "Okay"])
        return reply


# In[5]:


# build a ChatBot object
bot = ChatBot()
# start chatting
while True:
    # receive user input
    bot.user_input()
    # check whether to end chat
    if bot.end_chat:
        break
    # output bot response
    bot.bot_response()


# In[6]:


pickle.dump(model, open("model.pkl", "wb"))

# Loading model to compare the results
model = pickle.load(open("model.pkl", "rb"))


# In[7]:


pickle.dump(bot, open("bot.pkl", "wb"))
bot = pickle.load(open("bot.pkl", "rb"))


# In[ ]:
