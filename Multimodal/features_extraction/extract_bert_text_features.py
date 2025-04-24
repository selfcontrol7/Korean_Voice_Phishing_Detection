import json
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
import torch
from transformers import BertModel, BertTokenizer, pipeline
import multiprocessing as mp
import os

MODELS = {
    "kobert": "monologg/kobert",
    "krsbert": "snunlp/KR-SBERT-V40K-klueNLI-augSTS" #"jhgan/ko-sbert-nli-sts",
}

def load_tokenizer_model(model_choice, device):
    model_name = MODELS[model_choice]
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).eval().to(device)
    return tokenizer, model

def process_segment_cpu(seg):
    segment_id = seg["segment_id"]
    text = seg["text"]
    save_path = FEATURES_DIR / f"{segment_id}_{MODEL_CHOICE}.npy"

    if save_path.exists() and not OVERWRITE:
        seg[f"{MODEL_CHOICE}_text_path"] = str(save_path)
        return seg

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to("cpu")
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
    np.save(save_path, cls_embedding)

    seg[f"{MODEL_CHOICE}_text_path"] = str(save_path)
    return seg

def run_cpu_multiprocessing(segments, workers):
    with mp.Pool(workers, initializer=init_worker_cpu) as pool:
        results = list(tqdm(pool.imap(process_segment_cpu, segments), total=len(segments)))
    return results

def init_worker_cpu():
    global tokenizer, model
    tokenizer_local, model_local = load_tokenizer_model(MODEL_CHOICE, device="cpu")
    globals()["tokenizer"] = tokenizer_local
    globals()["model"] = model_local

def run_gpu_batch_processing(segments, batch_size=32):
    tokenizer, model = load_tokenizer_model(MODEL_CHOICE, device="cuda")
    results = []
    for i in tqdm(range(0, len(segments), batch_size), desc="GPU Batch Processing"):
        batch = segments[i:i+batch_size]
        texts = [seg["text"] for seg in batch]
        segment_ids = [seg["segment_id"] for seg in batch]

        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        for seg, emb, seg_id in zip(batch, cls_embeddings, segment_ids):
            save_path = FEATURES_DIR / f"{seg_id}_{MODEL_CHOICE}.npy"
            if not save_path.exists() or OVERWRITE:
                np.save(save_path, emb)
            seg[f"{MODEL_CHOICE}_text_path"] = str(save_path)
            results.append(seg)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--model", choices=["kobert", "krsbert"], default="kobert")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=int(os.cpu_count() * 0.8))
    args = parser.parse_args()

    global FEATURES_DIR, MODEL_CHOICE, OVERWRITE
    MODEL_CHOICE = args.model
    OVERWRITE = args.overwrite

    DATA_DIR = Path("data")
    FEATURES_DIR = Path(f"features/text_{MODEL_CHOICE}") # Directory to save features
    FEATURES_DIR.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

    segment_manifest_path = DATA_DIR / f"{args.split}_segment_manifest.jsonl" # Assuming this is the correct path
    # check if the segment manifest file exists
    if not segment_manifest_path.exists():
        raise FileNotFoundError(f"Segment manifest file not found: {segment_manifest_path}")
    output_manifest_path = DATA_DIR / f"{args.split}_segment_manifest_text_{MODEL_CHOICE}.jsonl" # Assuming this is the correct path

    with open(segment_manifest_path, encoding='utf-8') as f:
        segments = [json.loads(line) for line in f]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print("⚡ Using GPU with batch processing...")
        results = run_gpu_batch_processing(segments)
    else:
        print(f"🖥️ Using CPU with {args.workers} workers...")
        results = run_cpu_multiprocessing(segments, args.workers)

    with open(output_manifest_path, "w", encoding="utf-8") as fout:
        for item in results:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Saved {MODEL_CHOICE} features and updated manifest: {output_manifest_path}")

if __name__ == "__main__":
    main()
