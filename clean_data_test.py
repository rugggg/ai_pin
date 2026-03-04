import ray
import ray.data

ray.init()

ds = ray.data.from_huggingface("teknium/OpenHermes-2.5", split="train")

print(f"Original size: {ds.count()}")

BAD_SOURCES = ["camel_ai"]

def filter_sources(row):
    return row.get("source") not in BAD_SOURCES

ds = ds.filter(filter_sources)

def filter_chinese(row):
    for turn in row["conversations"]:
        if "Created Chinese" in turn["value"]:
            return False
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', turn["value"]))
        total_chars = len(turn["value"])
        if total_chars > 0 and chinese_chars / total_chars > 0.3:
            return False
    return True

ds = ds.filter(filter_chinese)

def filter_length(row):
    for turn in row["conversations"]:
        if turn["from"] == "gpt" and len(turn["value"]) > 2000:
            return False
    return True

def filter_too_short(row):
    for turn in row["conversations"]:
        if turn["from"] == "gpt" and len(turn["value"]) < 50:
            return False
    return True

ds = ds.filter(filter_too_short)

print(f"After filtering: {ds.count()}")

# convert back to HuggingFace dataset for training
clean_dataset = ds.to_huggingface()
print(clean_dataset)
