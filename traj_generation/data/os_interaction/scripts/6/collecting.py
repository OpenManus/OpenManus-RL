import json
import requests

with open("permission.json") as f:
    p = f.read()

f = open("collected.json", "a")
correct = 0
while correct < 1000:
    try:
        resp = requests.post(
            "http://45.41.95.10:10001/api/openai/chat-completion",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "user",
                     "content": "The topic is: linux permission questions.\ngenerate one like this:\n" + p[2:-1]},
                ]
            },
            timeout=60
        )
        entry = json.loads(resp.json()["choices"][0]["message"]["content"])
        f.write(json.dumps(entry))
        f.write(",\n")
        f.flush()
        correct += 1
    except Exception as e:
        print(e)
    if correct % 50 == 0:
        print(correct)
