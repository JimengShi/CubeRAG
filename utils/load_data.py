import json

def load_qa(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported file format. Use JSON or CSV.")
    


def load_corpus(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

        corpus = []
        for sample in data:
            corpus.append(sample['title'] + ":" + sample['text'])

    return corpus