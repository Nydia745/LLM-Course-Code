import gradio as gr

title = "GPT-J-6B"
description = "Gradio Demo for GPT-J 6B, a transformer model trained using Ben Wang's Mesh Transformer JAX. 'GPT-J' refers to the class of model, while '6B' represents the number of trainable parameters. To use it, simply add your text, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://github.com/kingoflolz/mesh-transformer-jax' target='_blank'>GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model</a></p>"

gr.Interface.load(
    "huggingface/EleutherAI/gpt-j-6B",
    inputs=gr.Textbox(lines=5, label="Input Text"),
    title=title,
    description=description,
    article=article,
).launch()

# local with spaces/
gr.Interface.load("spaces/abidlabs/remove-bg").launch()
gr.Interface.load(
    "spaces/abidlabs/remove-bg", inputs="webcam", title="Remove your webcam background!"
).launch()
