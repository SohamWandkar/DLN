"""
Fetches and audits SQuAD 2.0 samples for general reading comprehension and unanswerable question evaluation.
"""
import urllib.request
import json
import os

def fetch_squad_sample():
    dest_path = os.path.join(os.path.dirname(__file__), "raw", "squad2_sample.json")
    if os.path.exists(dest_path):
        print(f"SQuAD 2.0 sample already exists at {dest_path}")
        return

    print("Fetching SQuAD 2.0 dev set...")
    url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    with urllib.request.urlopen(req, timeout=30) as response:
        raw_data = json.loads(response.read().decode('utf-8'))
    
    extracted_items = []
    # Extract 30 answerable and 30 unanswerable examples across diverse topics
    count_ans = 0
    count_unans = 0
    
    for article in raw_data["data"]:
        topic = article["title"]
        for p in article["paragraphs"]:
            context = p["context"]
            for q in p["qas"]:
                is_impossible = q.get("is_impossible", False)
                if is_impossible and count_unans < 25:
                    extracted_items.append({
                        "id": q["id"],
                        "topic": topic,
                        "context": context,
                        "question": q["question"],
                        "is_answerable": False,
                        "answers": []
                    })
                    count_unans += 1
                elif not is_impossible and count_ans < 25:
                    answers = [a["text"] for a in q["answers"]]
                    extracted_items.append({
                        "id": q["id"],
                        "topic": topic,
                        "context": context,
                        "question": q["question"],
                        "is_answerable": True,
                        "answers": answers
                    })
                    count_ans += 1
            if count_ans >= 25 and count_unans >= 25:
                break
        if count_ans >= 25 and count_unans >= 25:
            break

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump({"total": len(extracted_items), "items": extracted_items}, f, indent=2)
    print(f"Successfully saved {len(extracted_items)} SQuAD 2.0 samples to {dest_path}")

if __name__ == "__main__":
    fetch_squad_sample()
