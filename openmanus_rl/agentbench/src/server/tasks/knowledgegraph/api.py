import json
import re
from pathlib import Path
from typing import Union

from .utils.logic_form_util import postprocess_raw_code, lisp_to_sparql, range_info

path = str(Path(__file__).parent.absolute())

with open(path + "/ontology/vocab.json") as f:
    vocab = json.load(f)
    attributes = vocab["attributes"]
    relations = vocab["relations"]


variable_relations_cache = {}
variable_attributes_cache = {}

class Variable:
    def __init__(self, type, program):
        self.type = type
        self.program = program
    def __hash__(self) -> int:
        return hash(self.program)
    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Variable):
            return self.program == __value.program
        else:
            return False
    def __repr__(self) -> str:
        return self.program

def final_execute(variable: Variable, sparql_executor):
    program = variable.program
    processed_code = postprocess_raw_code(program)
    sparql_query = lisp_to_sparql(processed_code)

    results = sparql_executor.execute_query(sparql_query)

    return results

def get_relations(variable: Union[Variable, str], sparql_executor):
    """
    Get all relations of a variable
    :param variable: here a variable is represented as its program derivation
    :return: a list of relations
    """
    if not isinstance(variable, Variable):
        if not re.match(r'^(m|f)\.[\w_]+$', variable):
            raise ValueError("get_relations: variable must be a variable or an entity")

    if isinstance(variable, Variable):
        program = variable.program

        processed_code = postprocess_raw_code(program)
        sparql_query = lisp_to_sparql(processed_code)
        clauses = sparql_query.split("\n")

        new_clauses = [clauses[0], "SELECT DISTINCT ?rel\nWHERE {\n?x ?rel ?obj .\n{"]
        new_clauses.extend(clauses[1:])
        new_clauses.append("}\n}")
        new_query = '\n'.join(new_clauses)
        out_relations = sparql_executor.execute_query(new_query)
    else: # variable is an entity
        out_relations = sparql_executor.get_out_relations(variable)

    out_relations = list(set(out_relations).intersection(set(relations)))

    # new_clauses = [clauses[0], "SELECT DISTINCT ?rel\nWHERE {\n?sub ?rel ?x .\n{"]
    # new_clauses.extend(clauses[1:])
    # new_clauses.append("}\n}")
    # new_query = '\n'.join(new_clauses)
    # in_relations = execute_query(new_query)

    rtn_str = f"Observation: [{', '.join(out_relations)}]"
    variable_relations_cache[variable] = out_relations

    return None, rtn_str


def get_neighbors(variable: Union[Variable, str], relation: str, sparql_executor):  # will create a new variable
    """
    Get all neighbors of a variable
    :param variable: a variable, here a variable is represented as its program derivation
    :param relation: a relation
    :return: a list of neighbors
    """
    if not isinstance(variable, Variable):
        if not re.match(r'^(m|f)\.[\w_]+$', variable):
            raise ValueError("get_neighbors: variable must be a variable or an entity")
    if not relation in variable_relations_cache[variable]:
        raise ValueError("get_neighbors: relation must be a relation of the variable")


    rtn_str = f"Observation: variable ##, which are instances of {range_info[relation]}"

    new_variable = Variable(range_info[relation],
                            f"(JOIN {relation + '_inv'} {variable.program if isinstance(variable, Variable) else variable})")

    return new_variable, rtn_str


def intersection(variable1: Variable, variable2: Variable, sparql_executor):  # will create a new variable
    """
    Get the intersection of two variables
    :param variable1: a variable
    :param variable2: a variable
    :return: a list of intersection
    """
    if variable1.type != variable2.type:
        raise ValueError("intersection: two variables must have the same type")

    if not isinstance(variable1, Variable) or not isinstance(variable2, Variable):
        raise ValueError("intersection: variable must be a variable")

    rtn_str = f"Observation: variable ##, which are instances of {variable1.type}"
    new_variable = Variable(variable1.type, f"(AND {variable1.program} {variable2.program})")
    return new_variable, rtn_str


def union(variable1: set, variable2: set, sparql_executor): # will create a new variable
    """
    Get the union of two variables
    :param variable1: a variable
    :param variable2: a variable
    :return: a list of union
    """
    if variable1.type != variable2.type:
        raise ValueError("union: two variables must have the same type")

    if not isinstance(variable1, Variable) or not isinstance(variable2, Variable):
        raise ValueError("union: variable must be a variable")

    rtn_str = f"Observation: variable ##, which are instances of {variable1.type}"
    new_variable = Variable(variable1.type, f"(OR {variable1.program} {variable2.program})")
    return new_variable, rtn_str


def count(variable: Variable, sparql_executor):
    """
    Count the number of a variable
    :param variable: a variable
    :return: the number of a variable
    """
    rtn_str = f"Observation: variable ##, which is a number"
    new_variable = Variable("type.int", f"(COUNT {variable.program})")
    return new_variable, rtn_str


def get_attributes(variable: Variable, sparql_executor):
    program = variable.program

    processed_code = postprocess_raw_code(program)
    sparql_query = lisp_to_sparql(processed_code)
    clauses = sparql_query.split("\n")

    new_clauses = [clauses[0], "SELECT DISTINCT ?rel\nWHERE {\n?x ?rel ?obj .\n{"]
    new_clauses.extend(clauses[1:])
    new_clauses.append("}\n}")
    new_query = '\n'.join(new_clauses)
    out_relations = sparql_executor.execute_query(new_query)

    out_relations = list(set(out_relations).intersection(set(attributes)))
    variable_attributes_cache[variable] = out_relations

    rtn_str = f"Observation: [{', '.join(out_relations)}]"

    return None, rtn_str



def argmax(variable: str, attribute: str, sparql_executor):
    """
    Get the argmax of a variable
    :param variable: a variable
    :param relation: a relation
    :return: the argmax of a variable
    """
    # program = f"(ARGMAX {variable} {attribute})"
    # processed_code = postprocess_raw_code(program)
    # sparql_query = lisp_to_sparql(processed_code)
    # answers = execute_query(sparql_query)
    if attribute not in variable_attributes_cache[variable]:
        raise ValueError("argmax: attribute must be an attribute of the variable")

    rtn_str = f"Observation: variable ##, which are instances of {variable.type}"
    new_variable = Variable(variable.type, f"(ARGMAX {variable.program} {attribute})")
    return new_variable, rtn_str


def argmin(variable: str, attribute: str, sparql_executor):
    """
    Get the argmin of a variable
    :param variable: a variable
    :param relation: a relation
    :return: the argmin of a variable
    """
    if attribute not in variable_attributes_cache[variable]:
        raise ValueError("argmin: attribute must be an attribute of the variable")

    rtn_str = f"Observation: variable ##, which are instances of {variable.type}"
    new_variable = Variable(variable.type, f"(ARGMIN {variable.program} {attribute})")
    return new_variable, rtn_str

