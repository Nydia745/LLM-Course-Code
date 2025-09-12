import gradio as gr
import numpy as np
from transformers import pipeline


def flip_text(x):
    return x[::-1]


demo = gr.Blocks()

with demo:
    gr.Markdown(
        """
    # Flip Text!
    Start typing below to see the output.
    """
    )
    input = gr.Textbox(placeholder="Flip this text")
    output = gr.Textbox()

    input.change(fn=flip_text, inputs=input, outputs=output)

demo.launch()

demo = gr.Blocks()


def flip_text(x):
    return x[::-1]


def flip_image(x):
    return np.fliplr(x)


with demo:
    gr.Markdown("Flip text or image files using this demo.")
    with gr.Tabs():
        with gr.TabItem("Flip Text"):
            with gr.Row():
                text_input = gr.Textbox()
                text_output = gr.Textbox()
            text_button = gr.Button("Flip")
        with gr.TabItem("Flip Image"):
            with gr.Row():
                image_input = gr.Image()
                image_output = gr.Image()
            image_button = gr.Button("Flip")

    text_button.click(flip_text, inputs=text_input, outputs=text_output)
    image_button.click(flip_image, inputs=image_input, outputs=image_output)

demo.launch()

api = gr.Interface.load("huggingface/EleutherAI/gpt-j-6B")


def complete_with_gpt(text):
    # Use the last 50 characters of the text as context
    return text[:-50] + api(text[-50:])


with gr.Blocks() as demo:
    textbox = gr.Textbox(placeholder="Type here and press enter...", lines=4)
    btn = gr.Button("Generate")

    btn.click(complete_with_gpt, textbox, textbox)

demo.launch()


asr = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")
classifier = pipeline("text-classification")


def speech_to_text(speech):
    text = asr(speech)["text"]
    return text


def text_to_sentiment(text):
    return classifier(text)[0]["label"]


demo = gr.Blocks()

with demo:
    audio_file = gr.Audio(type="filepath")
    text = gr.Textbox()
    label = gr.Label()

    b1 = gr.Button("Recognize Speech")
    b2 = gr.Button("Classify Sentiment")

    b1.click(speech_to_text, inputs=audio_file, outputs=text)
    b2.click(text_to_sentiment, inputs=text, outputs=label)

demo.launch()


def change_textbox(choice):
    if choice == "short":
        return gr.Textbox.update(lines=2, visible=True)
    elif choice == "long":
        return gr.Textbox.update(lines=8, visible=True)
    else:
        return gr.Textbox.update(visible=False)


with gr.Blocks() as block:
    radio = gr.Radio(
        ["short", "long", "none"], label="What kind of essay would you like to write?"
    )
    text = gr.Textbox(lines=2, interactive=True)

    radio.change(fn=change_textbox, inputs=radio, outputs=text)
    block.launch()
