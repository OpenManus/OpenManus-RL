from typing import List


def lisp_to_nested_expression(lisp_string: str) -> List:
    """
    Takes a logical form as a lisp string and returns a nested list representation of the lisp.
    For example, "(count (division first))" would get mapped to ['count', ['division', 'first']].
    """
    stack: List = []
    current_expression: List = []
    tokens = lisp_string.split()
    for token in tokens:
        while token[0] == '(':
            nested_expression: List = []
            current_expression.append(nested_expression)
            stack.append(current_expression)
            current_expression = nested_expression
            token = token[1:]
        current_expression.append(token.replace(')', ''))
        while token[-1] == ')':
            current_expression = stack.pop()
            token = token[:-1]
    return current_expression[0]

def expression_to_lisp(expression) -> str:
    rtn = '('
    for i, e in enumerate(expression):
        if isinstance(e, list):
            rtn += expression_to_lisp(e)
        else:
            rtn += e
        if i != len(expression) - 1:
            rtn += ' '

    rtn += ')'
    return rtn


def get_nesting_level(expression) -> int:
    max_sub = 0
    for item in expression:
        if isinstance(item, list):
            level = get_nesting_level(item)
            if level > max_sub:
                max_sub = level

    return 1 + max_sub



if __name__ == '__main__':
    lisp = '(AND common.topic (AND (JOIN common.topic.notable_types Comic Strip) (JOIN common.topic.notable_types Comic Strip)))'
    print(get_nesting_level(lisp_to_nested_expression(lisp)))

    print(expression_to_lisp(lisp_to_nested_expression(lisp)))
