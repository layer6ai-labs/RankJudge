"""Download and preprocess PubMedQA raw data into the standard paper format."""

import argparse
import json
import os

import pandas as pd
from huggingface_hub import hf_hub_download


def download(raw_dir, dataset_id, parquet_path, raw_file):
    """Download PubMedQA pqa_labeled parquet from Hugging Face and cache locally."""
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, raw_file)
    if os.path.exists(raw_path):
        print(f"Raw file already exists: {raw_path}", flush=True)
        return raw_path
    print(f"Downloading {dataset_id} ({parquet_path}) from Hugging Face...", flush=True)
    downloaded = hf_hub_download(
        repo_id=dataset_id,
        filename=parquet_path,
        repo_type="dataset",
        local_dir=raw_dir,
    )
    os.rename(downloaded, raw_path)
    subfolder = os.path.join(raw_dir, os.path.dirname(parquet_path))
    if os.path.isdir(subfolder):
        os.rmdir(subfolder)
    print(f"Saved -> {raw_path}", flush=True)
    return raw_path


def _format_context(row):
    """Build a single context block from a PubMedQA row."""
    parts = [f"Title: {row['question']}"]
    parts.append(f"\nContext:\n{chr(10).join(row['context']['contexts'])}")
    parts.append(f"\nReference Answer:\n{row['long_answer']}")
    return "\n".join(parts)


def preprocess(raw_path, output_path):
    """Convert raw PubMedQA parquet to unified {id, context} format."""
    df = pd.read_parquet(raw_path)
    papers = []
    for _, row in df.iterrows():
        papers.append({
            "id": str(row["pubid"]),
            "context": _format_context(row),
        })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(papers, f, indent=2)
    print(f"Converted {len(papers)} PubMedQA records -> {output_path}",  flush=True)
    return papers


def main():
    p = argparse.ArgumentParser(description="Download and preprocess PubMedQA dataset")
    p.add_argument("--dataset-id", default="qiaojin/PubMedQA",
                   help="Hugging Face dataset ID")
    p.add_argument("--parquet-path", default="pqa_labeled/train-00000-of-00001.parquet",
                   help="Path to parquet file within the dataset repo")
    p.add_argument("--raw-file", default="pqa_labeled.parquet",
                   help="Local filename for the downloaded parquet")
    p.add_argument("--output-file", default="med.json",
                   help="Output JSON filename")
    p.add_argument("--raw-dir", default="../data/raw",
                   help="Directory for raw downloaded data")
    p.add_argument("--out-dir", default="../data/input",
                   help="Directory for preprocessed output")
    args = p.parse_args()

    raw_path = download(args.raw_dir, args.dataset_id, args.parquet_path, args.raw_file)
    output_path = os.path.join(args.out_dir, args.output_file)
    preprocess(raw_path, output_path)


if __name__ == "__main__":
    main()
