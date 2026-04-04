import json
import csv
import random
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Create 200k headline dataset from two sources.")
    parser.add_argument("--dataset_a", default="./News_Category_Dataset_v3.json",
                        help="Path to HuffPost JSON lines file")
    parser.add_argument("--dataset_b", default="./abcnews-date-text.csv",
                        help="Path to ABC News CSV file")
    parser.add_argument("--output", default="rl_headlines_200k.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--count", type=int, default=200_000,
                        help="Number of headlines to output")
    parser.add_argument("--seed", type=int, default=81812376,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    headlines = []

    # Dataset A: JSON lines, one object per line with "headline" key
    print(f"Reading Dataset A: {args.dataset_a}")
    with open(args.dataset_a, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                headline = obj.get("headline", "").strip()
                if headline:
                    headlines.append(headline)
            except json.JSONDecodeError:
                continue
    print(f"  Loaded {len(headlines)} headlines from Dataset A")

    count_a = len(headlines)

    # Dataset B: CSV with columns publish_date, headline_text
    print(f"Reading Dataset B: {args.dataset_b}")
    with open(args.dataset_b, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            headline = row.get("headline_text", "").strip()
            if headline:
                headlines.append(headline)

    print(f"  Loaded {len(headlines) - count_a} headlines from Dataset B")
    print(f"Total headlines available: {len(headlines)}")

    if len(headlines) < args.count:
        raise ValueError(
            f"Not enough headlines: need {args.count}, have {len(headlines)}"
        )

    random.seed(args.seed)
    random.shuffle(headlines)
    selected = headlines[:args.count]

    output_path = Path(args.output)
    print(f"Writing {args.count} headlines to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for headline in selected:
            f.write(json.dumps({"input": headline}, ensure_ascii=False) + "\n")

    print("Done.")

if __name__ == "__main__":
    main()
