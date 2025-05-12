with open("llm_rl_scripts/wordle/vocab/wordle_official_400.txt", "r") as f:
    vocab = f.read().split("\n")

for word in vocab:
    if "l" in word and "a" in word and word[-1] == "t":
        print(word)