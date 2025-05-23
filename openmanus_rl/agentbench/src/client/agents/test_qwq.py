from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

reasoning_content = ""
content = ""

is_answering = False

completion = client.chat.completions.create(
    model="qwq-32b",
    messages=[
        {"role": "user", "content": "Which is larger, 9.9 or 9.11?"}
    ],
    stream=True,
    # Uncomment the following line to return token usage in the last chunk
    stream_options={
        "include_usage": True
    }
)

print("\n" + "=" * 20 + "reasoning content" + "=" * 20 + "\n")

for chunk in completion:
    # If chunk.choices is empty, print usage
    if not chunk.choices:
        print("\nUsage:")
        print(chunk.usage)
    else:
        delta = chunk.choices[0].delta
        # Print reasoning content
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
            print(delta.reasoning_content, end='', flush=True)
            reasoning_content += delta.reasoning_content
        else:
            if delta.content != "" and is_answering is False:
                print("\n" + "=" * 20 + "content" + "=" * 20 + "\n")
                is_answering = True
            # Print content
            print(delta.content, end='', flush=True)
            content += delta.content
