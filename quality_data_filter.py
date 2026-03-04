from transformers import pipeline
import ray

# Ray remote decorator turns this into a distributed actor
@ray.remote(num_gpus=0.5)  # share GPU across workers
class QualityScorer:
    def __init__(self):
        # load a small fast model for scoring
        self.scorer = pipeline(
            "text-classification",
            model="OpenAssistant/reward-model-deberta-v3-large-v2",
            device=0
        )
    
    def score(self, text):
        result = self.scorer(text[:512], truncation=True)
        return result[0]["score"]

def add_quality_score(row, scorer):
    # score the assistant response
    for turn in row["conversations"]:
        if turn["from"] == "gpt":
            score = ray.get(scorer.score.remote(turn["value"]))
            row["quality_score"] = score
            break
    return row

# create pool of scorer actors
scorers = [QualityScorer.remote() for _ in range(4)]

# score dataset in parallel
# then filter by score threshold
ds = ds.filter(lambda row: row.get("quality_score", 0) > 0.7)
