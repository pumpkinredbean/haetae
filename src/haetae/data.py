"""Map public datasets onto the three decision primitives.

Every loader yields normalized records:

    {
        "state": str,                     # the thing being judged
        "type": "choice" | "noul" | "score",
        "instructions": str,              # the question text
        "options": [str, ...],            # 2..255 candidate descriptions
        "label": int | None,              # hard gold index
        "soft": [float, ...] | None,      # soft target distribution
        "source": str,                    # dataset name for per-source eval
    }

Noul is stored as a 2-option choice whose option 0 means "yes".
Score options are ordered level descriptions, low to high.
"""

from __future__ import annotations

from typing import Iterator


def _download(url, dest):
    import os
    import urllib.request

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        urllib.request.urlretrieve(url, dest)
    return dest


def _row(state, type_, instructions, options, label=None, soft=None, source=""):
    return {
        "state": state,
        "type": type_,
        "instructions": instructions,
        "options": options,
        "label": label,
        "soft": soft,
        "source": source,
    }


def load_banking77(split="train", limit=None) -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset("mteb/banking77", split=split)
    names = [None] * 77
    for r in ds:
        names[r["label"]] = r["label_text"]
    opts = [n.replace("_", " ") for n in names]
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        yield _row(r["text"], "choice",
                   "Which customer-service intent does this message have?",
                   opts, label=r["label"], source="banking77")


def load_ag_news(split="train", limit=None) -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/ag_news", split=split)
    opts = ["world news", "sports", "business", "science and technology"]
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        yield _row(r["text"], "choice", "Which topic best fits this article?",
                   opts, label=r["label"], source="ag_news")


def load_mnli(split="train", limit=None) -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/multi_nli", split=split)
    opts = [
        "entailment: the hypothesis definitely follows from the premise",
        "neutral: the hypothesis may or may not follow",
        "contradiction: the hypothesis definitely does not follow",
    ]
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        if r["label"] == -1:
            continue
        yield _row(r["premise"], "choice",
                   f'Given the statement "{r["hypothesis"]}", how does it relate to the premise?',
                   opts, label=r["label"], source="mnli")


def load_boolq(split="train", limit=None) -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset("google/boolq", split=split)
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        yield _row(r["passage"], "noul", r["question"],
                   ["yes", "no"], label=0 if r["answer"] else 1,
                   source="boolq")


def load_sst5(split="train", limit=None) -> Iterator[dict]:
    """5-level ordinal sentiment -> score primitive."""
    from datasets import load_dataset

    ds = load_dataset("SetFit/sst5", split=split)
    opts = [
        "very negative", "negative", "neutral", "positive", "very positive",
    ]
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        yield _row(r["text"], "score", "Rate the sentiment of this text.",
                   opts, label=r["label"], source="sst5")


def load_civil_toxicity(split="train", limit=None) -> Iterator[dict]:
    """Soft-label noul: fraction of raters who marked the comment toxic."""
    from datasets import load_dataset

    ds = load_dataset("google/civil_comments", split=split)
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        p = float(r["toxicity"])
        yield _row(r["text"], "noul", "Is this comment toxic?",
                   ["yes", "no"], label=0 if p >= 0.5 else 1,
                   soft=[p, 1.0 - p], source="civil_toxicity")


def load_klue_ynat(split="train", limit=None) -> Iterator[dict]:
    """KLUE YNAT topic classification, from the official TSV."""
    import csv
    import os

    import json

    fname = "ynat-v1.1_train.json" if split == "train" else "ynat-v1.1_dev.json"
    path = _download(
        "https://github.com/KLUE-benchmark/KLUE/raw/main/klue_benchmark/ynat-v1.1/" + fname,
        os.path.join(os.path.dirname(__file__), "../../data", fname),
    )
    topics = ["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"]
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    for i, r in enumerate(rows):
        if limit and i >= limit:
            break
        yield _row(r["title"], "choice", "이 뉴스 기사의 주제는?",
                   topics, label=topics.index(r["label"]), source="klue_ynat")


def load_nsmc(split="train", limit=None) -> Iterator[dict]:
    """Naver sentiment movie corpus, from the official TSV."""
    import csv
    import os

    fname = "ratings_train.txt" if split == "train" else "ratings_test.txt"
    path = _download(
        "https://github.com/e9t/nsmc/raw/master/" + fname,
        os.path.join(os.path.dirname(__file__), "../../data", fname),
    )
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for i, r in enumerate(rows):
        if limit and i >= limit:
            break
        if not r["document"]:
            continue
        yield _row(r["document"], "noul", "이 리뷰는 긍정적인가?",
                   ["yes", "no"], label=0 if r["label"] == "1" else 1,
                   source="nsmc")


def load_massive(split="train", limit=None) -> Iterator[dict]:
    """MASSIVE intent routing, 60 intents."""
    from datasets import load_dataset

    ds = load_dataset("mteb/massive_intent", "en", split=split)
    seen = {}
    for r in ds:
        seen.setdefault(r["label_text"], len(seen))
    opts = [n.replace("_", " ") for n in seen]
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        yield _row(r["text"], "choice", "Which intent does this request have?",
                   opts, label=seen[r["label_text"]], source="massive")


def load_anli(split="train", limit=None) -> Iterator[dict]:
    """Adversarial NLI — the hard version of entailment."""
    from datasets import load_dataset

    if split == "train":
        split = "train_r1"
    ds = load_dataset("facebook/anli", split=split)
    opts = [
        "entailment: the hypothesis definitely follows from the premise",
        "neutral: the hypothesis may or may not follow",
        "contradiction: the hypothesis definitely does not follow",
    ]
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        yield _row(r["premise"], "choice",
                   f'Given the statement "{r["hypothesis"]}", how does it relate to the premise?',
                   opts, label=r["label"], source="anli")


def load_helpsteer2(split="train", limit=None) -> Iterator[dict]:
    """Five ordinal Score questions over one prompt/response state."""
    from datasets import load_dataset

    ds = load_dataset("nvidia/HelpSteer2", split=split)
    levels = ["0 (lowest)", "1", "2", "3", "4 (highest)"]
    attrs = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        state = f"PROMPT: {r['prompt']}\n\nRESPONSE: {r['response']}"
        import hashlib
        parent = "helpsteer2:prompt:" + hashlib.sha256(
            r["prompt"].encode("utf-8")).hexdigest()[:16]
        for a in attrs:
            rec = _row(state, "score",
                       f"Rate the {a} of the response to the prompt.",
                       levels, label=int(r[a]), source=f"helpsteer2_{a}")
            rec["parent"] = parent
            yield rec


def load_arc(split="train", limit=None) -> Iterator[dict]:
    """ARC: choice with real answer texts, not label names."""
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        texts = r["choices"]["text"]
        keys = r["choices"]["label"]
        if r["answerKey"] not in keys:
            continue
        yield _row(r["question"], "choice",
                   "Which option correctly answers the question?",
                   list(texts), label=keys.index(r["answerKey"]),
                   source="arc")


def load_emotion(split="train", limit=None) -> Iterator[dict]:
    """SetFit/emotion: 6-way emotion, same set public benchmarks use."""
    from datasets import load_dataset

    ds = load_dataset("SetFit/emotion", split=split)
    seen = {}
    for r in ds:
        seen.setdefault(r["label_text"], len(seen))
    names = list(seen)
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        yield _row(r["text"], "choice", "Which emotion best fits this text?",
                   names, label=seen[r["label_text"]], source="emotion")



LOADERS = {
    "banking77": load_banking77,
    "ag_news": load_ag_news,
    "mnli": load_mnli,
    "boolq": load_boolq,
    "sst5": load_sst5,
    "civil_toxicity": load_civil_toxicity,
    "klue_ynat": load_klue_ynat,
    "nsmc": load_nsmc,
    "massive": load_massive,
    "anli": load_anli,
    "helpsteer2": load_helpsteer2,
    "arc": load_arc,
    "emotion": load_emotion,
}
