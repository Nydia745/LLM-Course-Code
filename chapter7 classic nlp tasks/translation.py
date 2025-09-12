from datasets import load_dataset

raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
raw_datasets

# DatasetDict({
#     train: Dataset({
#         features: ['id', 'translation'],
#         num_rows: 210173
#     })
# })
split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
split_datasets

# DatasetDict({
#     train: Dataset({
#         features: ['id', 'translation'],
#         num_rows: 189155
#     })
#     test: Dataset({
#         features: ['id', 'translation'],
#         num_rows: 21018
#     })
# })
split_datasets["validation"] = split_datasets.pop("test")
split_datasets["train"][1]["translation"]

# {'en': 'Default to expanded threads',
#  'fr': 'Par défaut, développer les fils de discussion'}
from transformers import pipeline

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
translator = pipeline("translation", model=model_checkpoint)
translator("Default to expanded threads")

# [{'translation_text': 'Par défaut pour les threads élargis'}]
split_datasets["train"][172]["translation"]

# {'en': 'Unable to import %1 using the OFX importer plugin. This file is not the correct format.',
#  'fr': "Impossible d'importer %1 en utilisant le module d'extension d'importation OFX. Ce fichier n'a pas un format correct."}
translator(
    "Unable to import %1 using the OFX importer plugin. This file is not the correct format."
)

# [{'translation_text': "Impossible d'importer %1 en utilisant le plugin d'importateur OFX. Ce fichier n'est pas le bon format."}]
from transformers import AutoTokenizer

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
en_sentence = split_datasets["train"][1]["translation"]["en"]
fr_sentence = split_datasets["train"][1]["translation"]["fr"]

inputs = tokenizer(en_sentence, text_target=fr_sentence)
inputs

# {'input_ids': [47591, 12, 9842, 19634, 9, 0], 'attention_mask': [1, 1, 1, 1, 1, 1], 'labels': [577, 5891, 2, 3184, 16, 2542, 5, 1710, 0]}
wrong_targets = tokenizer(fr_sentence)
print(tokenizer.convert_ids_to_tokens(wrong_targets["input_ids"]))
print(tokenizer.convert_ids_to_tokens(inputs["labels"]))

# ['▁Par', '▁dé', 'f', 'aut', ',', '▁dé', 've', 'lop', 'per', '▁les', '▁fil', 's', '▁de', '▁discussion', '</s>']
# ['▁Par', '▁défaut', ',', '▁développer', '▁les', '▁fils', '▁de', '▁discussion', '</s>']
max_length = 128


def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs


tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 3)])
batch.keys()

# dict_keys(['attention_mask', 'input_ids', 'labels', 'decoder_input_ids'])
batch["labels"]

# tensor([[  577,  5891,     2,  3184,    16,  2542,     5,  1710,     0,  -100,
#           -100,  -100,  -100,  -100,  -100,  -100],
#         [ 1211,     3,    49,  9409,  1211,     3, 29140,   817,  3124,   817,
#            550,  7032,  5821,  7907, 12649,     0]])
batch["decoder_input_ids"]

# tensor([[59513,   577,  5891,     2,  3184,    16,  2542,     5,  1710,     0,
#          59513, 59513, 59513, 59513, 59513, 59513],
#         [59513,  1211,     3,    49,  9409,  1211,     3, 29140,   817,  3124,
#            817,   550,  7032,  5821,  7907, 12649]])
for i in range(1, 3):
    print(tokenized_datasets["train"][i]["labels"])

# [577, 5891, 2, 3184, 16, 2542, 5, 1710, 0]
# [1211, 3, 49, 9409, 1211, 3, 29140, 817, 3124, 817, 550, 7032, 5821, 7907, 12649, 0]
import evaluate

metric = evaluate.load("sacrebleu")
predictions = [
    "This plugin lets you translate web pages between several languages automatically."
]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
metric.compute(predictions=predictions, references=references)

# {'score': 46.750469682990165,
#  'counts': [11, 6, 4, 3],
#  'totals': [12, 11, 10, 9],
#  'precisions': [91.67, 54.54, 40.0, 33.33],
#  'bp': 0.9200444146293233,
#  'sys_len': 12,
#  'ref_len': 13}
predictions = ["This This This This"]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
metric.compute(predictions=predictions, references=references)

# {'score': 1.683602693167689,
#  'counts': [1, 0, 0, 0],
#  'totals': [4, 3, 2, 1],
#  'precisions': [25.0, 16.67, 12.5, 12.5],
#  'bp': 0.10539922456186433,
#  'sys_len': 4,
#  'ref_len': 13}
predictions = ["This plugin"]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
metric.compute(predictions=predictions, references=references)

# {'score': 0.0,
#  'counts': [2, 1, 0, 0],
#  'totals': [2, 1, 0, 0],
#  'precisions': [100.0, 100.0, 0.0, 0.0],
#  'bp': 0.004086771438464067,
#  'sys_len': 2,
#  'ref_len': 13}
import numpy as np


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}


from huggingface_hub import notebook_login

notebook_login()
from transformers import Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    f"marian-finetuned-kde4-en-to-fr",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
)
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.evaluate(max_length=max_length)

# {'eval_loss': 1.6964408159255981,
#  'eval_bleu': 39.26865061007616,
#  'eval_runtime': 965.8884,
#  'eval_samples_per_second': 21.76,
#  'eval_steps_per_second': 0.341}
trainer.train()
trainer.evaluate(max_length=max_length)

# {'eval_loss': 0.8558505773544312,
#  'eval_bleu': 52.94161337775576,
#  'eval_runtime': 714.2576,
#  'eval_samples_per_second': 29.426,
#  'eval_steps_per_second': 0.461,
#  'epoch': 3.0}
trainer.push_to_hub(tags="translation", commit_message="Training complete")

# 'https://huggingface.co/sgugger/marian-finetuned-kde4-en-to-fr/commit/3601d621e3baae2bc63d3311452535f8f58f6ef3'
from torch.utils.data import DataLoader

tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], collate_fn=data_collator, batch_size=8
)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
from transformers import get_scheduler

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
from huggingface_hub import Repository, get_full_repo_name

model_name = "marian-finetuned-kde4-en-to-fr-accelerate"
repo_name = get_full_repo_name(model_name)
repo_name

# 'sgugger/marian-finetuned-kde4-en-to-fr-accelerate'
output_dir = "marian-finetuned-kde4-en-to-fr-accelerate"
repo = Repository(output_dir, clone_from=repo_name)


def postprocess(predictions, labels):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    return decoded_preds, decoded_labels


import torch
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,
            )
        labels = batch["labels"]

        # Necessary to pad predictions and labels for being gathered
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(generated_tokens)
        labels_gathered = accelerator.gather(labels)

        decoded_preds, decoded_labels = postprocess(
            predictions_gathered, labels_gathered
        )
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    results = metric.compute()
    print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )

# epoch 0, BLEU score: 53.47
# epoch 1, BLEU score: 54.24
# epoch 2, BLEU score: 54.44
from transformers import pipeline

# Replace this with your own checkpoint
model_checkpoint = "huggingface-course/marian-finetuned-kde4-en-to-fr"
translator = pipeline("translation", model=model_checkpoint)
translator("Default to expanded threads")

# [{'translation_text': 'Par défaut, développer les fils de discussion'}]
translator(
    "Unable to import %1 using the OFX importer plugin. This file is not the correct format."
)

# [{'translation_text': "Impossible d'importer %1 en utilisant le module externe d'importation OFX. Ce fichier n'est pas le bon format."}]
