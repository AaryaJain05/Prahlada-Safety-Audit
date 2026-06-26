import json
from pathlib import Path


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "harmful_romanized_hindi_data_5000.json"
    output_path = base_dir / "harmful_romanized_hindi_only_data_5000.json"

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    converted = []
    for item in data:
        converted.append(
            {
                "Romanized_Hindi": item.get("Romanized_Hindi"),
                "safe_completion_romanized_hi": item.get("safe_completion_romanized_hi"),
                "harmful_completion_romanized_hi": item.get("harmful_completion_romanized_hi"),
                "Category": item.get("Category"),
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
