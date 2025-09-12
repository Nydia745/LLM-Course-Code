import gradio as gr
import numpy as np
from transformers import pipeline


def reverse_audio(audio):
    sr, data = audio
    reversed_audio = (sr, np.flipud(data))
    return reversed_audio


mic = gr.Audio(sources=["microphone"], type="numpy", label="Speak here...")
demo1 = gr.Interface(reverse_audio, mic, "audio").launch()

notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def generate_tone(note, octave, duration):
    sr = 48000
    a4_freq, tones_from_a4 = 440, 12 * (octave - 4) + (note - 9)
    frequency = a4_freq * 2 ** (tones_from_a4 / 12)
    duration = int(duration)
    audio = np.linspace(0, duration, duration * sr)
    audio = (20000 * np.sin(audio * (2 * np.pi * frequency))).astype(np.int16)
    return (sr, audio)


gr.Interface(
    generate_tone,
    [
        gr.Dropdown(notes, type="index"),
        gr.Slider(minimum=4, maximum=6, step=1),
        gr.Number(value=1, label="Duration in seconds"),
    ],
    "audio",
).launch()

model = pipeline("automatic-speech-recognition")


def transcribe_audio(audio):
    transcription = model(audio)["text"]
    return transcription


gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
).launch()
