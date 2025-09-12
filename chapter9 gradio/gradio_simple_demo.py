import gradio as gr
from transformers import pipeline


def greet(name):
    return "Hello " + name


demo1 = gr.Interface(fn=greet, inputs="text", outputs="text")


def greet(name):
    return "Hello " + name


# We instantiate the Textbox class
textbox = gr.Textbox(label="Type your name here:", placeholder="John Doe", lines=2)

demo2 = gr.Interface(fn=greet, inputs=textbox, outputs="text")

model = pipeline("text-generation")


def predict(prompt):
    completion = model(prompt)[0]["generated_text"]
    return completion


gr.Interface(fn=predict, inputs="text", outputs="text").launch()
