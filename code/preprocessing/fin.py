"""Download and preprocess S&P 500 10-K filings into the standard paper format."""

import argparse
import json
import os

import pandas as pd
from huggingface_hub import hf_hub_download


PARQUET_FILES = [f"data/{year}.parquet" for year in range(2010, 2023)]

SECTION_BUDGETS = {
    "item_1": 3000,
    "item_1A": 3000,
    "item_7": 6000,
}

SECTION_LABELS = {
    "item_1": "Business Description",
    "item_1A": "Risk Factors",
    "item_7": "Management Discussion & Analysis",
}


def download(raw_dir, dataset_id):
    """Download all yearly parquet files from Hugging Face and cache locally."""
    os.makedirs(raw_dir, exist_ok=True)
    paths = []
    for parquet_path in PARQUET_FILES:
        local_name = os.path.basename(parquet_path)
        raw_path = os.path.join(raw_dir, local_name)
        if os.path.exists(raw_path):
            paths.append(raw_path)
            continue
        print(f"Downloading {parquet_path} from {dataset_id}...", flush=True)
        downloaded = hf_hub_download(
            repo_id=dataset_id,
            filename=parquet_path,
            repo_type="dataset",
            local_dir=raw_dir,
        )
        # Move from nested data/ subfolder to raw_dir root
        if downloaded != raw_path:
            os.rename(downloaded, raw_path)
        paths.append(raw_path)
    # Clean up empty data/ subfolder if created by hf_hub_download
    subfolder = os.path.join(raw_dir, "data")
    if os.path.isdir(subfolder) and not os.listdir(subfolder):
        os.rmdir(subfolder)
    print(f"Downloaded {len(paths)} parquet files -> {raw_dir}", flush=True)
    return paths


def _truncate_section(text, max_chars):
    """Truncate text to the nearest sentence boundary before max_chars."""
    if not text or len(text) <= max_chars:
        return text or ""
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars // 2:
        return truncated[:last_period + 1]
    return truncated


def _format_context(row):
    """Build a single context block from a 10-K filing row."""
    date_str = str(row['date'])[:10]  # strip time component from Timestamp
    parts = [f"Company: {row['company']}"]
    parts.append(f"Filing Date: {date_str}")
    parts.append(f"Industry (SIC): {row['sic']}")
    if pd.notna(row.get("mkt_cap")):
        parts.append(f"Market Cap: {row['mkt_cap']}")
    if pd.notna(row.get("ret")):
        parts.append(f"Annual Return: {row['ret']}")

    for col, label in SECTION_LABELS.items():
        text = row.get(col)
        if text and isinstance(text, str) and text.strip():
            truncated = _truncate_section(text.strip(), SECTION_BUDGETS[col])
            parts.append(f"\n{label}:\n{truncated}")

    return "\n".join(parts)


def _total_section_text(row):
    """Return total character count across the three selected sections."""
    total = 0
    for col in SECTION_LABELS:
        text = row.get(col)
        if text and isinstance(text, str):
            total += len(text.strip())
    return total


def preprocess(raw_paths, output_path, min_context_chars, n_samples, seed):
    """Convert raw 10-K parquet files to unified {id, context} format."""
    dfs = [pd.read_parquet(p) for p in raw_paths]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} total filings", flush=True)

    # Filter out rows with insufficient text
    df = df[df.apply(lambda r: _total_section_text(r) >= min_context_chars, axis=1)]
    print(f"After filtering (>= {min_context_chars} chars): {len(df)} filings", flush=True)

    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=seed)
        print(f"Sampled {n_samples} filings", flush=True)

    papers = []
    for _, row in df.iterrows():
        date_str = str(row['date'])[:10]
        papers.append({
            "id": f"{row['cik']}_{date_str}",
            "context": _format_context(row),
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(papers, f, indent=2)
    print(f"Converted {len(papers)} 10-K filings -> {output_path}", flush=True)
    return papers


def main():
    p = argparse.ArgumentParser(description="Download and preprocess S&P 500 10-K dataset")
    p.add_argument("--dataset-id", default="jlohding/sp500-edgar-10k",
                   help="Hugging Face dataset ID")
    p.add_argument("--output-file", default="fin.json",
                   help="Output JSON filename")
    p.add_argument("--raw-dir", default="../data/raw",
                   help="Directory for raw downloaded data")
    p.add_argument("--out-dir", default="../data/input",
                   help="Directory for preprocessed output")
    p.add_argument("--min-context-chars", type=int, default=2000,
                   help="Minimum total chars across sections to keep a filing")
    p.add_argument("--n-samples", type=int, default=None,
                   help="Optional cap on number of filings to include")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for sampling")
    args = p.parse_args()

    raw_paths = download(args.raw_dir, args.dataset_id)
    output_path = os.path.join(args.out_dir, args.output_file)
    preprocess(raw_paths, output_path, args.min_context_chars, args.n_samples, args.seed)


if __name__ == "__main__":
    main()
