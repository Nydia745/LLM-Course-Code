from datasets import load_dataset

squad_it_dataset = load_dataset("json", data_files="SQuAD_it-train.json", field="data")
squad_it_dataset

# DatasetDict({
#     train: Dataset({
#         features: ['title', 'paragraphs'],
#         num_rows: 442
#     })
# })
squad_it_dataset["train"][0]

# {
#     "title": "Terremoto del Sichuan del 2008",
#     "paragraphs": [
#         {
#             "context": "Il terremoto del Sichuan del 2008 o il terremoto...",
#             "qas": [
#                 {
#                     "answers": [{"answer_start": 29, "text": "2008"}],
#                     "id": "56cdca7862d2951400fa6826",
#                     "question": "In quale anno si è verificato il terremoto nel Sichuan?",
#                 },
#                 ...
#             ],
#         },
#         ...
#     ],
# }
data_files = {"train": "SQuAD_it-train.json", "test": "SQuAD_it-test.json"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
squad_it_dataset

# DatasetDict({
#     train: Dataset({
#         features: ['title', 'paragraphs'],
#         num_rows: 442
#     })
#     test: Dataset({
#         features: ['title', 'paragraphs'],
#         num_rows: 48
#     })
# })

# support automatic decompression of the input files
data_files = {"train": "SQuAD_it-train.json.gz", "test": "SQuAD_it-test.json.gz"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")

# Loading a remote dataset
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
