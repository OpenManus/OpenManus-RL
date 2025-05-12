import re
import random


def extract_solution(solution_str):
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    else:
        return None

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    if final_answer is not None:
        try:
            int_final_answer = int(final_answer)
        except ValueError:
            final_answer = None
    return final_answer


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"--------------------------------")
        print(f"Ground truth: {ground_truth} | Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        if do_print:
            print(f"No answer found")
        return 0
    else:
        if int(answer) == int(ground_truth):
            if do_print:
                print(f"Correct answer: {answer}")
            return score
        else:
            if do_print:
                print(f"Incorrect answer {answer} | Ground truth: {ground_truth}")
            return format_score
