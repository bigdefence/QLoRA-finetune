import huggingface_hub
from datasets import Dataset
huggingface_hub.login()

json_file='custom_json_file'
dataset = Dataset.from_json(json_file)
dataset.push_to_hub('huggingface_dataset 저장소')

