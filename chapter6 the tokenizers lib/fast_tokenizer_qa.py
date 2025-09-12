import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

question_answerer = pipeline("question-answering")
context = """
🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch, and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
question = "Which deep learning libraries back 🤗 Transformers?"
question_answerer(question=question, context=context)

# {'score': 0.97773,
#  'start': 78,
#  'end': 105,
#  'answer': 'Jax, PyTorch and TensorFlow'}
long_context = """
🤗 Transformers: State of the Art NLP

🤗 Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction,
question answering, summarization, translation, text generation and more in over 100 languages.
Its aim is to make cutting-edge NLP easier to use for everyone.

🤗 Transformers provides APIs to quickly download and use those pretrained models on a given text, fine-tune them on your own datasets and
then share them with the community on our model hub. At the same time, each python module defining an architecture is fully standalone and
can be modified to enable quick research experiments.

Why should I use transformers?

1. Easy-to-use state-of-the-art models:
  - High performance on NLU and NLG tasks.
  - Low barrier to entry for educators and practitioners.
  - Few user-facing abstractions with just three classes to learn.
  - A unified API for using all our pretrained models.
  - Lower compute costs, smaller carbon footprint:

2. Researchers can share trained models instead of always retraining.
  - Practitioners can reduce compute time and production costs.
  - Dozens of architectures with over 10,000 pretrained models, some in more than 100 languages.

3. Choose the right framework for every part of a model's lifetime:
  - Train state-of-the-art models in 3 lines of code.
  - Move a single model between TF2.0/PyTorch frameworks at will.
  - Seamlessly pick the right framework for training, evaluation and production.

4. Easily customize a model or an example to your needs:
  - We provide examples for each architecture to reproduce the results published by its original authors.
  - Model internals are exposed as consistently as possible.
  - Model files can be used independently of the library for quick experiments.

🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
question_answerer(question=question, context=long_context)

# {'score': 0.97149,
#  'start': 1892,
#  'end': 1919,
#  'answer': 'Jax, PyTorch and TensorFlow'}

model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)
start_logits = outputs.start_logits
end_logits = outputs.
























print(start_logits.shape, end_logits.shape)
# torch.Size([1, 66]) torch.Size([1, 66])

sequence_ids = inputs.sequence_ids()
# Mask everything apart from the tokens of the context
mask = [i != 1 for i in sequence_ids]
# Unmask the [CLS] token
mask[0] = False
mask = torch.tensor(mask)[None]

start_logits[mask] = -10000
end_logits[mask] = -10000
start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)[0]
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)es[None, :]
scores = torch.triu(scores)
max_index = scores.argmax().item()
start_index = max_index // scores.shape[1]
end_index = max_index % scores.shape[1]
print(scores[start_index, end_index])

# 0.97773
inputs_with_offsets = tokenizer(question, context, return_offsets_mapping=True)
offsets = inputs_with_offsets["offset_mapping"]

start_char, _ = offsets[start_index]
_, end_char = offsets[end_index]
answer = context[start_char:end_char]
result = {
    "answer": answer,
    "start": start_char,
    "end": end_char,
    "score": scores[start_index, end_index],
}
print(result)

# {'answer': 'Jax, PyTorch and TensorFlow',
#  'start': 78,
#  'end': 105,
#  'score': 0.97773}
inputs = tokenizer(question, long_context)
print(len(inputs["input_ids"]))

# 461
inputs = tokenizer(question, long_context, max_length=384, truncation="only_second")
print(tokenizer.decode(inputs["input_ids"]))

# """
# [CLS] Which deep learning libraries back [UNK] Transformers? [SEP] [UNK] Transformers : State of the Art NLP
#
# [UNK] Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction,
# question answering, summarization, translation, text generation and more in over 100 languages.
# Its aim is to make cutting-edge NLP easier to use for everyone.
#
# [UNK] Transformers provides APIs to quickly download and use those pretrained models on a given text, fine-tune them on your own datasets and
# then share them with the community on our model hub. At the same time, each python module defining an architecture is fully standalone and
# can be modified to enable quick research experiments.
#
# Why should I use transformers?
#
# 1. Easy-to-use state-of-the-art models:
#   - High performance on NLU and NLG tasks.
#   - Low barrier to entry for educators and practitioners.
#   - Few user-facing abstractions with just three classes to learn.
#   - A unified API for using all our pretrained models.
#   - Lower compute costs, smaller carbon footprint:
#
# 2. Researchers can share trained models instead of always retraining.
#   - Practitioners can reduce compute time and production costs.
#   - Dozens of architectures with over 10,000 pretrained models, some in more than 100 languages.
#
# 3. Choose the right framework for every part of a model's lifetime:
#   - Train state-of-the-art models in 3 lines of code.
#   - Move a single model between TF2.0/PyTorch frameworks at will.
#   - Seamlessly pick the right framework for training, evaluation and production.
#
# 4. Easily customize a model or an example to your needs:
#   - We provide examples for each architecture to reproduce the results published by its original authors.
#   - Model internal [SEP]
# """
sentence = "This sentence is not too long but we are going to split it anyway."
inputs = tokenizer(
    sentence, truncation=True, return_overflowing_tokens=True, max_length=6, stride=2
)

for ids in inputs["input_ids"]:
    print(tokenizer.decode(ids))

# '[CLS] This sentence is not [SEP]'
# '[CLS] is not too long [SEP]'
# '[CLS] too long but we [SEP]'
# '[CLS] but we are going [SEP]'
# '[CLS] are going to split [SEP]'
# '[CLS] to split it anyway [SEP]'
# '[CLS] it anyway. [SEP]'
print(inputs.keys())

# dict_keys(['input_ids', 'attention_mask', 'overflow_to_sample_mapping'])
print(inputs["overflow_to_sample_mapping"])

# [0, 0, 0, 0, 0, 0, 0]
sentences = [
    "This sentence is not too long but we are going to split it anyway.",
    "This sentence is shorter but will still get split.",
]
inputs = tokenizer(
    sentences, truncation=True, return_overflowing_tokens=True, max_length=6, stride=2
)

print(inputs["overflow_to_sample_mapping"])

# [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
inputs = tokenizer(
    question,
    long_context,
    stride=128,
    max_length=384,
    padding="longest",
    truncation="only_second",
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)
_ = inputs.pop("overflow_to_sample_mapping")
offsets = inputs.pop("offset_mapping")

inputs = inputs.convert_to_tensors("pt")
print(inputs["input_ids"].shape)

# torch.Size([2, 384])
outputs = model(**inputs)

start_logits = outputs.start_logits
end_logits = outputs.end_logits
print(start_logits.shape, end_logits.shape)

# torch.Size([2, 384]) torch.Size([2, 384])
sequence_ids = inputs.sequence_ids()
# Mask everything apart from the tokens of the context
mask = [i != 1 for i in sequence_ids]
# Unmask the [CLS] token
mask[0] = False
# Mask all the [PAD] tokens
mask = torch.logical_or(torch.tensor(mask)[None], (inputs["attention_mask"] == 0))

start_logits[mask] = -10000
end_logits[mask] = -10000
start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)
candidates = []
for start_probs, end_probs in zip(start_probabilities, end_probabilities):
    scores = start_probs[:, None] * end_probs[None, :]
    idx = torch.triu(scores).argmax().item()

    start_idx = idx // scores.shape[1]
    end_idx = idx % scores.shape[1]
    score = scores[start_idx, end_idx].item()
    candidates.append((start_idx, end_idx, score))

print(candidates)

# [(0, 18, 0.33867), (173, 184, 0.97149)]
for candidate, offset in zip(candidates, offsets):
    start_token, end_token, score = candidate
    start_char, _ = offset[start_token]
    _, end_char = offset[end_token]
    answer = long_context[start_char:end_char]
    result = {"answer": answer, "start": start_char, "end": end_char, "score": score}
    print(result)

# {'answer': '\n🤗 Transformers: State of the Art NLP', 'start': 0, 'end': 37, 'score': 0.33867}
# {'answer': 'Jax, PyTorch and TensorFlow', 'start': 1892, 'end': 1919, 'score': 0.97149}
