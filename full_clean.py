"""
OpenHermes 2.5 Data Cleaning Pipeline
Filters noisy samples from teknium/OpenHermes-2.5 using Ray for parallel processing.

Filters applied:
1. Remove known noisy sources
2. Remove Chinese bilingual artifacts
3. Remove overly verbose responses (>2000 chars)
4. Remove very short responses (<50 chars)
5. Remove samples with no assistant response

Usage:
    pip install ray datasets
    python clean_openhermes.py
"""

import re
import ray
import ray.data
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────

DATASET_ID = "teknium/OpenHermes-2.5"
OUTPUT_PATH = "./openhermes_clean"
SAMPLE_SIZE = None  # set to an int e.g. 100000 to limit, None = full dataset

MAX_RESPONSE_LENGTH = 2000   # filter responses longer than this
MIN_RESPONSE_LENGTH = 50     # filter responses shorter than this
MAX_CHINESE_RATIO   = 0.3    # filter if >30% chinese characters

# sources known to produce noisy/weird outputs - add more as you discover them
BAD_SOURCES = [
    "camel_ai",
]

# ── Filters ───────────────────────────────────────────────────────────────────

def filter_bad_sources(row):
    """Remove samples from known noisy source datasets."""
    source = row.get("source") or ""
    return source not in BAD_SOURCES


def filter_chinese_artifacts(row):
    """
    Remove samples containing Chinese bilingual artifacts or
    responses that are predominantly Chinese.
    """
    conversations = row.get("conversations") or []
    for turn in conversations:
        value = turn.get("value") or ""

        # catch the specific OpenHermes bilingual artifact pattern
        if "Created Chinese" in value:
            return False
        if "### Created Chinese" in value:
            return False

        # catch responses that are mostly chinese characters
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', value))
        total_chars   = len(value)
        if total_chars > 0 and (chinese_chars / total_chars) > MAX_CHINESE_RATIO:
            return False

    return True


def filter_response_length(row):
    """
    Remove samples where the assistant response is too long (verbose)
    or too short (likely low quality / empty).
    """
    conversations = row.get("conversations") or []
    has_assistant = False

    for turn in conversations:
        if turn.get("from") != "gpt":
            continue

        value = turn.get("value") or ""
        has_assistant = True

        if len(value) > MAX_RESPONSE_LENGTH:
            return False
        if len(value) < MIN_RESPONSE_LENGTH:
            return False

    # drop samples with no assistant turn at all
    return has_assistant


def filter_meta_commentary(row):
    """
    Remove responses that contain common low-quality meta patterns
    e.g. 'Note: I have written this response in a ...' or
    '(Note: This is a fictional ...'
    """
    meta_patterns = [
        r"\(Note:",
        r"\[Note:",
        r"Note: I have",
        r"Note: This response",
        r"Note: The above",
        r"Please note that I",
        r"As an AI language model,",
        r"As an AI, I",
    ]
    combined = re.compile("|".join(meta_patterns), re.IGNORECASE)

    conversations = row.get("conversations") or []
    for turn in conversations:
        if turn.get("from") != "gpt":
            continue
        value = turn.get("value") or ""
        if combined.search(value):
            return False

    return True


# ── Pipeline ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("OpenHermes 2.5 Cleaning Pipeline")
    print("=" * 60)

    # init Ray locally - uses all available CPUs
    ray.init(ignore_reinit_error=True)

    # load dataset via HuggingFace then hand off to Ray
    print(f"\nLoading {DATASET_ID}...")
    hf_ds = load_dataset(DATASET_ID, split="train")

    if SAMPLE_SIZE:
        hf_ds = hf_ds.shuffle(seed=42).select(range(SAMPLE_SIZE))

    original_count = len(hf_ds)
    print(f"Loaded {original_count:,} samples")

    # convert to Ray Dataset for parallel filtering
    ds = ray.data.from_huggingface(hf_ds)

    # ── apply filters in order of cost (cheapest first) ──

    print("\nApplying filters...")

    ds = ds.filter(filter_bad_sources)
    after_sources = ds.count()
    print(f"  After source filter:       {after_sources:,}  "
          f"(removed {original_count - after_sources:,})")

    ds = ds.filter(filter_chinese_artifacts)
    after_chinese = ds.count()
    print(f"  After Chinese filter:      {after_chinese:,}  "
          f"(removed {after_sources - after_chinese:,})")

    ds = ds.filter(filter_response_length)
    after_length = ds.count()
    print(f"  After length filter:       {after_length:,}  "
          f"(removed {after_chinese - after_length:,})")

    ds = ds.filter(filter_meta_commentary)
    after_meta = ds.count()
    print(f"  After meta commentary:     {after_meta:,}  "
          f"(removed {after_length - after_meta:,})")

    # ── summary ──

    final_count = after_meta
    removed     = original_count - final_count
    retention   = (final_count / original_count) * 100

    print(f"\n{'=' * 60}")
    print(f"Original:  {original_count:,}")
    print(f"Cleaned:   {final_count:,}")
    print(f"Removed:   {removed:,}  ({100 - retention:.1f}%)")
    print(f"Retained:  {retention:.1f}%")
    print(f"{'=' * 60}")

    # ── source breakdown of what remains ──
    print("\nTop sources in cleaned dataset:")
    hf_clean = ds.to_huggingface()

    from collections import Counter
    source_counts = Counter(
        row["source"] for row in hf_clean
        if row.get("source")
    )
    for source, count in source_counts.most_common(15):
        pct = (count / final_count) * 100
        print(f"  {source:<40} {count:>7,}  ({pct:.1f}%)")

    # ── save ──
    print(f"\nSaving cleaned dataset to {OUTPUT_PATH}...")
    hf_clean.save_to_disk(OUTPUT_PATH)
    print("Done!")
    print(f"\nTo use in training:")
    print(f"  from datasets import load_from_disk")
    print(f"  dataset = load_from_disk('{OUTPUT_PATH}')")

    ray.shutdown()


if __name__ == "__main__":
    main()
