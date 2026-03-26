import json


with open('data/eval.json', 'r') as f:
    data = json.load(f)


filtered_data = []
for item in data:
    if "null" in item["conversations"][-1]["value"].lower():
        continue
    filtered_data.append(item)

print(len(filtered_data))
with open('data/eval_filtered.json', 'w') as f:
    json.dump(filtered_data, f, indent=4)