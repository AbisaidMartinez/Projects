# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 00:23:45 2026

@author: qbo28
"""

import ollama
import streamlit as st

#print(ollama.list())
        
st.title("Local Chatbot with Ollama")

def respuesta_modelo():
    stream = ollama.chat(
        model=st.session_state["modelo"],
        messages=st.session_state["messages"], 
        #[{'role': 'user', 'content': '¿Por qué es azul el cielo?'}],
        stream = True)
    
    # Mostrar la respuesta generada por el modelo
    for chunk in stream:    
        yield chunk['message']['content']

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    
modelos = [modelo["model"] for modelo in ollama.list()["models"]]
st.session_state["modelo"] = st.selectbox("Select the model", modelos)
        
    # def respuesta_modelo():
    #     stream = ollama.chat(
    #         model=st.session_state["modelo"],
    #         messages=st.session_state["messages"], #[{'role': 'user', 'content': '¿Por qué es azul el cielo?'}],
    #         stream = True)
        
    #     # Mostrar la respuesta generada por el modelo
    #     for chunk in stream:    
    #         yield chunk['message']['content']
            
for mensaje in st.session_state["messages"]:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])
            
if prompt := st.chat_input("Introduce your message..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        message = st.write_stream(respuesta_modelo())
        st.session_state["messages"].append({"role": "user", "content": message})
                    
                    