import random

import gradio as gr
import requests
import tensorflow as tf


def chat(message, history):
    history = history or []
    if message.startswith("How many"):
        response = random.randint(1, 10)
    elif message.startswith("How"):
        response = random.choice(["Great", "Good", "Okay", "Bad"])
    elif message.startswith("Where"):
        response = random.choice(["Here", "There", "Somewhere"])
    else:
        response = "I don't know"
    history.append((message, response))
    return history, history


iface = gr.Interface(
    chat,
    ["text", "state"],
    ["chatbot", "state"],
    allow_screenshot=False,
    allow_flagging="never",
)
iface.launch()


inception_net = tf.keras.applications.MobileNetV2()  # load the model

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")


def classify_image(inp):
    inp = inp.reshape((-1, 224, 224, 3))
    inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    prediction = inception_net.predict(inp).flatten()
    return {labels[i]: float(prediction[i]) for i in range(1000)}


image = gr.Image(shape=(224, 224))
label = gr.Label(num_top_classes=3)

title = "Gradio Image Classifiction + Interpretation Example"
gr.Interface(
    fn=classify_image,
    inputs=image,
    outputs=label,
    interpretation="default",
    title=title,
).launch()
