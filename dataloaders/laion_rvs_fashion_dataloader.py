from datasets import load_dataset

dataset = load_dataset(
    "Slep/LAION-RVS-Fashion",
    streaming=True
)

train = dataset["train"]

for example in train:
    print(example)
    break
