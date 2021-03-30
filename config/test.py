import json

with open('pretrain-btm-base-8gpu.json', 'r', encoding='utf-8') as f:
    content = f.read()
    a = json.loads(content)
    print(a)
