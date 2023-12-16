import json

def prepare_dataset():
    
    # wget https://huggingface.co/datasets/flytech/llama-python-codes-30k/raw/main/llama-python-codes-30k-cleaned.jsonl
    with open('~/Downloads/Llama Python Codes 30k.jsonl', 'r') as f:
        data = [json.loads(line)["output"] for line in f]

    with open('data/train.txt', 'w') as f:
        f.write("\n".join(data))


if __name__ == "__main__":
    prepare_dataset()
