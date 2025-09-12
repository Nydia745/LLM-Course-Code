import torch
from huggingface_hub import notebook_login
from transformers import AutoModel, AutoTokenizer, BertModel

model = AutoModel.from_pretrained("bert-base-cased")

# if you know the type of model you want to use, you can use the class that defines its architecture directly
model = BertModel.from_pretrained("bert-base-cased")


model.save_pretrained("directory_on_my_computer")

model = AutoModel.from_pretrained("directory_on_my_computer")

notebook_login()

model.push_to_hub("my-awesome-model")

model = AutoModel.from_pretrained("your-username/my-awesome-model")

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

encoded_input = tokenizer("Hello, I'm a single sentence!")
print(encoded_input)

# {'input_ids': [101, 8667, 117, 1000, 1045, 1005, 1049, 2235, 17662, 12172, 1012, 102],
#  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

# input_ids: numerical representations of your tokens
# token_type_ids: these tell the model which part of the input is sentence A and which is sentence B
# attention_mask: this indicates which tokens should be attended to and which should not
tokenizer.decode(encoded_input["input_ids"])

# "[CLS] Hello, I'm a single sentence! [SEP]"
encoded_input = tokenizer("How are you?", "I'm fine, thank you!")
print(encoded_input)

# {'input_ids': [[101, 1731, 1132, 1128, 136, 102], [101, 1045, 1005, 1049, 2503, 117, 5763, 1128, 136, 102]],
#  'token_type_ids': [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
#  'attention_mask': [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
encoded_input = tokenizer("How are you?", "I'm fine, thank you!", return_tensors="pt")
print(encoded_input)

# {'input_ids': tensor([[  101,  1731,  1132,  1128,   136,   102],
#          [  101,  1045,  1005,  1049,  2503,   117,  5763,  1128,   136,   102]]),
#  'token_type_ids': tensor([[0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
#  'attention_mask': tensor([[1, 1, 1, 1, 1, 1],
#          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
encoded_input = tokenizer(
    ["How are you?", "I'm fine, thank you!"], padding=True, return_tensors="pt"
)
print(encoded_input)

# {'input_ids': tensor([[  101,  1731,  1132,  1128,   136,   102,     0,     0,     0,     0],
#          [  101,  1045,  1005,  1049,  2503,   117,  5763,  1128,   136,   102]]),
#  'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
#  'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
#          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
encoded_input = tokenizer(
    "This is a very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very long sentence.",
    truncation=True,
)
print(encoded_input["input_ids"])

# [101, 1188, 1110, 170, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1179, 5650, 119, 102]
encoded_input = tokenizer(
    ["How are you?", "I'm fine, thank you!"],
    padding=True,
    truncation=True,
    max_length=5,
    return_tensors="pt",
)
print(encoded_input)

# {'input_ids': tensor([[  101,  1731,  1132,  1128,   102],
#          [  101,  1045,  1005,  1049,   102]]),
#  'token_type_ids': tensor([[0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0]]),
#  'attention_mask': tensor([[1, 1, 1, 1, 1],
#          [1, 1, 1, 1, 1]])}
encoded_input = tokenizer("How are you?")
print(encoded_input["input_ids"])
tokenizer.decode(encoded_input["input_ids"])

# [101, 1731, 1132, 1128, 136, 102]
# '[CLS] How are you? [SEP]'
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
encoded_sequences = [
    [
        101,
        1045,
        1005,
        2310,
        2042,
        3403,
        2005,
        1037,
        17662,
        12172,
        2607,
        2026,
        2878,
        2166,
        1012,
        102,
    ],
    [101, 1045, 5223, 2023, 2061, 2172, 999, 102],
]

model_inputs = torch.tensor(encoded_sequences)
output = model(model_inputs)
