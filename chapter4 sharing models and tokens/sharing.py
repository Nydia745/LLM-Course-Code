from huggingface_hub import (  # User management; Repository creation and management; And some methods to retrieve/change information about the content
    Repository,
    create_repo,
    delete_file,
    delete_repo,
    list_datasets,
    list_metrics,
    list_models,
    list_repo_files,
    login,
    logout,
    notebook_login,
    update_repo_visibility,
    upload_file,
    whoami,
)
from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainingArguments

notebook_login()

training_args = TrainingArguments(
    "bert-finetuned-mrpc", save_strategy="epoch", push_to_hub=True
)

checkpoint = "camembert-base"

model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model.push_to_hub("dummy-model")
tokenizer.push_to_hub("dummy-model")
tokenizer.push_to_hub("dummy-model", organization="huggingface")
tokenizer.push_to_hub(
    "dummy-model", organization="huggingface", use_auth_token="<TOKEN>"
)

create_repo("dummy-model")

create_repo("dummy-model", organization="huggingface")

upload_file(
    "<path_to_file>/config.json",
    path_in_repo="config.json",
    repo_id="<namespace>/dummy-model",
)

repo = Repository("<path_to_dummy_folder>", clone_from="<namespace>/dummy-model")
repo.git_pull()
repo.git_add()
repo.git_commit()
repo.git_push()
repo.git_tag()
repo.git_pull()
model.save_pretrained("<path_to_dummy_folder>")
tokenizer.save_pretrained("<path_to_dummy_folder>")
repo.git_add()
repo.git_commit("Add model and tokenizer files")
repo.git_push()

checkpoint = "camembert-base"

model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Do whatever with the model, train it, fine-tune it...

model.save_pretrained("<path_to_dummy_folder>")
tokenizer.save_pretrained("<path_to_dummy_folder>")
