import json, os, sys

def process_file(path):
    meta_json = json.load(open(os.path.join(path, 'meta.json')))
    replay = json.load(open(os.path.join(path, 'replay.json')))
    win = [1, 0] if meta_json["winner"] == '0' else [0, 1]
    damage = [0, 0]
    takedown = [0, 0]
    try_times = [0, 0]
    full_play = [1, 1]
    round_num = len(replay) - 1
    
    last_round = replay[-2]
    
    if 'errors' in last_round:
        error = last_round['errors'][0]
        error_player = error['player']
        full_play[error_player] = 0
    
    if last_round['players'][0]['id'] == 0:
        for fish_0 in last_round['players'][0]['fight_fish']:
            damage[1] += 400 - max(fish_0['hp'], 0)
            takedown[1] += 1 if fish_0['hp'] <= 0 else 0
        for fish_1 in last_round['players'][1]['fight_fish']:
            damage[0] += 400 - max(fish_1['hp'], 0)
            takedown[0] += 1 if fish_1['hp'] <= 0 else 0
    else:
        for fish_0 in last_round['players'][0]['fight_fish']:
            damage[0] += 400 - max(fish_0['hp'], 0)
            takedown[0] += 1 if fish_0['hp'] <= 0 else 0
        for fish_1 in last_round['players'][1]['fight_fish']:
            damage[1] += 400 - max(fish_1['hp'], 0)
            takedown[1] += 1 if fish_1['hp'] <= 0 else 0
    
    if os.path.exists(os.path.join(path, 'thinking_process_0.jsonl')):
        for data in open(os.path.join(path, 'thinking_process_0.jsonl')).readlines():
            data = json.loads(data)
            try_times[0] += data['try_times']
    if os.path.exists(os.path.join(path, 'guess_process_0.jsonl')):
        for data in open(os.path.join(path, 'guess_process_0.jsonl')).readlines():
            data = json.loads(data)
            try_times[0] += data['try_times']
    
    if os.path.exists(os.path.join(path, 'thinking_process_1.jsonl')):
        for data in open(os.path.join(path, 'thinking_process_1.jsonl')).readlines():
            data = json.loads(data)
            try_times[1] += data['try_times']
    if os.path.exists(os.path.join(path, 'guess_process_1.jsonl')):
        for data in open(os.path.join(path, 'guess_process_1.jsonl')).readlines():
            data = json.loads(data)
            try_times[1] += data['try_times']
    
    return full_play, try_times, takedown, damage, win, round_num

def calculate(result_dir, agent):
    total_full_play = [0, 0]
    total_try_times = [0, 0]
    total_takedown = [0, 0]
    total_damage = [0, 0]
    total_win = [0, 0]
    total_round_num = 0
    test_times = 0
    
    for filename in os.listdir(result_dir):
        try:
            full_play, try_times, takedown, damage, win, round_num = process_file(os.path.join(result_dir, filename))
        except:
            continue
        
        total_full_play[0] += full_play[0]
        total_full_play[1] += full_play[1]
        
        total_try_times[0] += try_times[0]
        total_try_times[1] += try_times[1]
        
        total_takedown[0] += takedown[0]
        total_takedown[1] += takedown[1]
        
        total_damage[0] += damage[0]
        total_damage[1] += damage[1]
        
        total_win[0] += win[0]
        total_win[1] += win[1]
        
        total_round_num += round_num
        
        test_times += 1
    return {
        'full_play': total_full_play[agent],
        'try_times': total_try_times[agent],
        'takedown': total_takedown[agent],
        'damage': total_damage[agent],
        'win_round': total_win[agent],
        'round_num': total_round_num,
        'test_times': test_times
    }
            

if __name__ == '__main__':
    print(json.dumps(calculate(sys.argv[1], int(sys.argv[2]))))