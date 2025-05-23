from tqdm import tqdm
from typing import List
import re
import threading
import jsonlines
import yaml
import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def bleu_score(reference, candidate):
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()

    smoothie = SmoothingFunction().method4
    score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)
    return score

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob

def process_action(action, choices, limit=0.01, to_print=False):
    if to_print:
        print("preprocess action: ", action)
    match = re.search("ACTION:(.*)", action)
    if match:
        action = match.group(1)
    else:
        return False

    action = action.strip().lower().split("\n")[0]
    if not choices:
        return action
    if action in choices:
        return action
    try:
        bleus = [bleu_score(choice, action) for choice in choices]
        max_index = np.argmax(np.array(bleus))
        max_score = bleus[max_index]
        if max_score > limit:
            if to_print:
                print("processed action: ", choices[max_index], " score: ", max_score)
            return choices[max_index]
    except Exception as e:
        print("encounter exception: ", e)
        print("choices: ", choices)
        print("action: ", action)
    return action

def load_prompts(prompts_file):
    with open(prompts_file, 'r') as f:
        d = json.load(f)
        f.close()
    return d

def load_config(config_file):
    with open(config_file) as reader:
        config = yaml.safe_load(reader)
    return config