import csv
import json

# Input and output paths
csv_file_path = 'train.csv'
output_jsonl_path = 'llava_formatted_dataset.jsonl'

# Read CSV and convert to LLaVA format
with open(csv_file_path, 'r', encoding='utf-8') as csv_file, open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
    reader = csv.DictReader(csv_file)
    for idx, row in enumerate(reader):
        data = {
            "id": f"sample_{idx}",
            "image": row["Figure_path"],
            "conversations": [
                {"from": "human", "value": row["Question"]},
                {"from": "gpt", "value": row["Answer"]}
            ]
        }
        jsonl_file.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"✅ Converted dataset saved to {output_jsonl_path}")
