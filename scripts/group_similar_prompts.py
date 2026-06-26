"""
Group similar prompts by embedding similarity for manual review.

Usage:
    pip install sentence-transformers scikit-learn numpy
    
"""

import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_dataset(path):
    with open(path, "r") as f:
        return json.load(f)


def encode_prompts(data):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    prompts = [item["English"] for item in data]
    print(f"Encoding {len(prompts)} prompts...")
    embeddings = model.encode(prompts, show_progress_bar=True)
    return embeddings


def group_similar(data, embeddings, threshold=0.92):
    sims = cosine_similarity(embeddings)

    visited = set()
    groups = []

    for i in range(len(data)):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        for j in range(i + 1, len(data)):
            if j not in visited and sims[i][j] >= threshold:
                group.append(j)
                visited.add(j)
        if len(group) > 1:
            groups.append(group)

    # Singletons (no duplicates found)
    singletons = [i for i in range(len(data)) if not any(i in g for g in groups)]

    return groups, singletons


def print_groups(data, groups, singletons, output_path=None):
    lines = []

    lines.append(f"{'='*80}")
    lines.append(f"TOTAL ITEMS: {len(data)}")
    lines.append(f"DUPLICATE GROUPS: {len(groups)}")
    lines.append(f"ITEMS IN GROUPS: {sum(len(g) for g in groups)}")
    lines.append(f"UNIQUE (no duplicates): {len(singletons)}")
    lines.append(f"{'='*80}\n")

    for idx, group in enumerate(groups):
        lines.append(f"\n{'─'*80}")
        lines.append(f"GROUP {idx+1} ({len(group)} similar prompts)")
        lines.append(f"{'─'*80}")
        for rank, i in enumerate(group):
            item = data[i]
            lines.append(f"\n  [{rank+1}] index={i}  category={item.get('category', 'N/A')}")
            lines.append(f"  PROMPT: {item['English']}")
            lines.append(f"  SAFE:   {item.get('safe_completion_en', 'N/A')}")
            lines.append(f"  HARMFUL:{item.get('harmful_completion_en', 'N/A')}")

    output = "\n".join(lines)
    print(output)

    if output_path:
        with open(output_path, "w") as f:
            f.write(output)
        print(f"\nSaved to {output_path}")


def export_groups_json(data, groups, singletons, output_path):
    result = {
        "summary": {
            "total": len(data),
            "duplicate_groups": len(groups),
            "items_in_groups": sum(len(g) for g in groups),
            "unique_singletons": len(singletons),
        },
        "groups": [
            {
                "group_id": idx + 1,
                "item_count": len(group),
                "items": [data[i] | {"original_index": i} for i in group],
            }
            for idx, group in enumerate(groups)
        ],
        "singletons": [data[i] | {"original_index": i} for i in singletons],
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Exported groups JSON to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to dataset JSON")
    parser.add_argument("--threshold", type=float, default=0.92, help="Similarity threshold (default 0.92)")
    parser.add_argument(
        "--txt-out",
        default="similar_groups_{threshold:.2f}.txt",
        help="Output text report path; may include {threshold} placeholder",
    )
    parser.add_argument(
        "--json-out",
        default="similar_groups_{threshold:.2f}.json",
        help="Output JSON groups path; may include {threshold} placeholder",
    )
    args = parser.parse_args()

    data = load_dataset(args.input)
    embeddings = encode_prompts(data)

    threshold = args.threshold
    while True:
        groups, singletons = group_similar(data, embeddings, threshold=threshold)

        txt_out = args.txt_out.format(threshold=threshold)
        json_out = args.json_out.format(threshold=threshold)

        print_groups(data, groups, singletons, output_path=txt_out)
        export_groups_json(data, groups, singletons, output_path=json_out)

        response = input(
            "\nEnter a new threshold to retune (0.0-1.0), or press Enter to exit: "
        ).strip()
        if response == "":
            break
        try:
            threshold = float(response)
            if threshold < 0 or threshold > 1:
                raise ValueError
        except ValueError:
            print("Invalid threshold. Please enter a number between 0.0 and 1.0.")
            threshold = args.threshold
            continue
        print(f"Retuning with threshold={threshold:.2f}")
