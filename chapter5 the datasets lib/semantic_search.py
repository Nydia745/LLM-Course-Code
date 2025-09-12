import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModel, AutoTokenizer

issues_dataset = load_dataset("lewtun/github-issues", split="train")
issues_dataset

# Dataset({
#     features: ['url', 'repository_url', 'labels_url', 'comments_url', 'events_url', 'html_url', 'id', 'node_id', 'number', 'title', 'user', 'labels', 'state', 'locked', 'assignee', 'assignees', 'milestone', 'comments', 'created_at', 'updated_at', 'closed_at', 'author_association', 'active_lock_reason', 'pull_request', 'body', 'performed_via_github_app', 'is_pull_request'],
#     num_rows: 2855
# })
issues_dataset = issues_dataset.filter(
    lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0)
)
issues_dataset

# Dataset({
#     features: ['url', 'repository_url', 'labels_url', 'comments_url', 'events_url', 'html_url', 'id', 'node_id', 'number', 'title', 'user', 'labels', 'state', 'locked', 'assignee', 'assignees', 'milestone', 'comments', 'created_at', 'updated_at', 'closed_at', 'author_association', 'active_lock_reason', 'pull_request', 'body', 'performed_via_github_app', 'is_pull_request'],
#     num_rows: 771
# })
columns = issues_dataset.column_names
columns_to_keep = ["title", "body", "html_url", "comments"]
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
issues_dataset = issues_dataset.remove_columns(columns_to_remove)
issues_dataset

# Dataset({
#     features: ['html_url', 'title', 'comments', 'body'],
#     num_rows: 771
# })
issues_dataset.set_format("pandas")
df = issues_dataset[:]
df["comments"][0].tolist()

# ['the bug code locate in ：\r\n    if data_args.task_name is not None:\r\n        # Downloading and loading a dataset from the hub.\r\n        datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)',
#  'Hi @jinec,\r\n\r\nFrom time to time we get this kind of `ConnectionError` coming from the github.com website: https://raw.githubusercontent.com\r\n\r\nNormally, it should work if you wait a little and then retry.\r\n\r\nCould you please confirm if the problem persists?',
#  'cannot connect，even by Web browser，please check that  there is some  problems。',
#  'I can access https://raw.githubusercontent.com/huggingface/datasets/1.7.0/datasets/glue/glue.py without problem...']
comments_df = df.explode("comments", ignore_index=True)
comments_df.head(4)

comments_dataset = Dataset.from_pandas(comments_df)
comments_dataset

# Dataset({
#     features: ['html_url', 'title', 'comments', 'body'],
#     num_rows: 2842
# })
comments_dataset = comments_dataset.map(
    lambda x: {"comment_length": len(x["comments"].split())}
)
comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)
comments_dataset


# Dataset({
#     features: ['html_url', 'title', 'comments', 'body', 'comment_length'],
#     num_rows: 2098
# })
def concatenate_text(examples):
    return {
        "text": examples["title"]
        + " \n "
        + examples["body"]
        + " \n "
        + examples["comments"]
    }


comments_dataset = comments_dataset.map(concatenate_text)

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

device = torch.device("cuda")
model.to(device)


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]


def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


embedding = get_embeddings(comments_dataset["text"][0])
embedding.shape

# torch.Size([1, 768])
embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)
embeddings_dataset.add_faiss_index(column="embeddings")
question = "How can I load a dataset offline?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()
question_embedding.shape

# torch.Size([1, 768])
scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
)

samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores
samples_df.sort_values("scores", ascending=False, inplace=True)
for _, row in samples_df.iterrows():
    print(f"COMMENT: {row.comments}")
    print(f"SCORE: {row.scores}")
    print(f"TITLE: {row.title}")
    print(f"URL: {row.html_url}")
    print("=" * 50)
    print()

# """
# COMMENT: Requiring online connection is a deal breaker in some cases unfortunately so it'd be great if offline mode is added similar to how `transformers` loads models offline fine.
#
# @mandubian's second bullet point suggests that there's a workaround allowing you to use your offline (custom?) dataset with `datasets`. Could you please elaborate on how that should look like?
# SCORE: 25.505046844482422
# TITLE: Discussion using datasets in offline mode
# URL: https://github.com/huggingface/datasets/issues/824
# ==================================================
#
# COMMENT: The local dataset builders (csv, text , json and pandas) are now part of the `datasets` package since #1726 :)
# You can now use them offline
# \`\`\`python
# datasets = load_dataset("text", data_files=data_files)
# \`\`\`
#
# We'll do a new release soon
# SCORE: 24.555509567260742
# TITLE: Discussion using datasets in offline mode
# URL: https://github.com/huggingface/datasets/issues/824
# ==================================================
#
# COMMENT: I opened a PR that allows to reload modules that have already been loaded once even if there's no internet.
#
# Let me know if you know other ways that can make the offline mode experience better. I'd be happy to add them :)
#
# I already note the "freeze" modules option, to prevent local modules updates. It would be a cool feature.
#
# ----------
#
# > @mandubian's second bullet point suggests that there's a workaround allowing you to use your offline (custom?) dataset with `datasets`. Could you please elaborate on how that should look like?
#
# Indeed `load_dataset` allows to load remote dataset script (squad, glue, etc.) but also you own local ones.
# For example if you have a dataset script at `./my_dataset/my_dataset.py` then you can do
# \`\`\`python
# load_dataset("./my_dataset")
# \`\`\`
# and the dataset script will generate your dataset once and for all.
#
# ----------
#
# About I'm looking into having `csv`, `json`, `text`, `pandas` dataset builders already included in the `datasets` package, so that they are available offline by default, as opposed to the other datasets that require the script to be downloaded.
# cf #1724
# SCORE: 24.14896583557129
# TITLE: Discussion using datasets in offline mode
# URL: https://github.com/huggingface/datasets/issues/824
# ==================================================
#
# COMMENT: > here is my way to load a dataset offline, but it **requires** an online machine
# >
# > 1. (online machine)
# >
# > ```
# >
# > import datasets
# >
# > data = datasets.load_dataset(...)
# >
# > data.save_to_disk(/YOUR/DATASET/DIR)
# >
# > ```
# >
# > 2. copy the dir from online to the offline machine
# >
# > 3. (offline machine)
# >
# > ```
# >
# > import datasets
# >
# > data = datasets.load_from_disk(/SAVED/DATA/DIR)
# >
# > ```
# >
# >
# >
# > HTH.
#
#
# SCORE: 22.893993377685547
# TITLE: Discussion using datasets in offline mode
# URL: https://github.com/huggingface/datasets/issues/824
# ==================================================
#
# COMMENT: here is my way to load a dataset offline, but it **requires** an online machine
# 1. (online machine)
# \`\`\`
# import datasets
# data = datasets.load_dataset(...)
# data.save_to_disk(/YOUR/DATASET/DIR)
# \`\`\`
# 2. copy the dir from online to the offline machine
# 3. (offline machine)
# \`\`\`
# import datasets
# data = datasets.load_from_disk(/SAVED/DATA/DIR)
# \`\`\`
#
# HTH.
# SCORE: 22.406635284423828
# TITLE: Discussion using datasets in offline mode
# URL: https://github.com/huggingface/datasets/issues/824
# ==================================================
# """
