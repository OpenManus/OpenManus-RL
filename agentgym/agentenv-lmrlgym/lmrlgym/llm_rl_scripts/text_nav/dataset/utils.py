from typing import Generator, Dict, List, Tuple, Union
import re
from LLM_RL.environment import Text, TextHistory, TextTrajectory


def data_stream(fp: Iterator) -> Generator[Dict, None, None]:
    """Generator for reading data json files."""
    for line in fp:
        line = line.strip()
        if line == '':
            continue
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError:
            print(f'Error parsing json line:\n{line}')
            continue
        for i in range(0, len(data['text_history']), 2):
            in_text = "\n\n".join(data['text_history'][:i+1])
            out_text = data['text_history'][i+1]
            yield {'in_text':in_text, 'out_text':out_text}

