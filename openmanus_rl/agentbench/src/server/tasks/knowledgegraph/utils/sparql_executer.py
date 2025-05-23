from typing import List, Tuple, Union
from SPARQLWrapper import SPARQLWrapper, JSON
import json
import urllib
from pathlib import Path
from tqdm import tqdm


path = str(Path(__file__).parent.absolute())

with open(path + '/../ontology/fb_roles', 'r') as f:
    contents = f.readlines()

roles = set()
for line in contents:
    fields = line.split()
    roles.add(fields[1])


class SparqlExecuter:
    def __init__(self, url: Union[str, None]=None):
        self.sparql = SPARQLWrapper(url or "http://164.107.116.56:3093/sparql")
        self.sparql.setReturnFormat(JSON)

    def execute_query(self, query: str) -> List[str]:
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        rtn = []
        for result in results['results']['bindings']:
            assert len(result) == 1  # only select one variable
            for var in result:
                rtn.append(result[var]['value'].replace('http://rdf.freebase.com/ns/', '').replace("-08:00", ''))

        return rtn


    def execute_unary(self, type: str) -> List[str]:
        query = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?value) WHERE {
        SELECT DISTINCT ?x0  WHERE {
        """
                '?x0 :type.object.type :' + type + '. '
                                                    """
        }
        }
        """)
        # # print(query)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        rtn = []
        for result in results['results']['bindings']:
            rtn.append(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

        return rtn


    def execute_binary(self, relation: str) -> List[Tuple[str, str]]:
        query = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT DISTINCT ?x0 ?x1 WHERE {
        """
                '?x0 :' + relation + ' ?x1. '
                                    """
        }
        """)
        # # print(query)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        rtn = []
        for result in results['results']['bindings']:
            rtn.append((result['x0']['value'], result['x1']['value']))

        return rtn


    def is_intersectant(self, derivation1: tuple, derivation2: str):
        if len(derivation1[1]) > 3 or len(derivation2[1]) > 3:
            return False

        if len(derivation1) == 2:
            clause1 = derivation1[0] + ' ' + ' / '.join(derivation1[1]) + ' ?x. \n'
        elif len(derivation1) == 3:
            clause1 = '?y ' + ' / '.join(derivation1[1]) + ' ?x. \n' + f'FILTER (?y {derivation1[2]} {derivation1[0]}) . \n'

        if len(derivation2) == 2:
            clause2 = derivation2[0] + ' ' + ' / '.join(derivation2[1]) + ' ?x. \n'
        elif len(derivation2) == 3:
            clause2 = '?y ' + ' / '.join(derivation2[1]) + ' ?x. \n' + f'FILTER (?y {derivation2[2]} {derivation2[0]}) . \n'

        query = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/> 
            ASK {
            """
                + clause1
                + clause2 +
                """
    }
    """)
        # print(query)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        rtn = results['boolean']
        return rtn


    def entity_type_connected(self, entity: str, type: str):
        query = ("""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/> 
                ASK {
                """
                + ':' + entity + '  !(<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>|:type.object.type) '
                                '/ :type.object.type :' + type +
                """
    }
    """)
        # print(query)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        rtn = results['boolean']
        return rtn


    def entity_type_connected_2hop(self, entity: str, type: str):
        query = ("""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/> 
                ASK {
                """
                + ':' + entity + '  !(<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>|:type.object.type) / '
                                ' !(<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>|:type.object.type)'
                                '/ :type.object.type :' + type +
                """
    }
    """)
        # print(query)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        rtn = results['boolean']
        return rtn


    def get_in_attributes(self, value: str):
        in_attributes = set()

        query1 = ("""
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX : <http://rdf.freebase.com/ns/> 
                    SELECT (?x0 AS ?value) WHERE {
                    SELECT DISTINCT ?x0  WHERE {
                    """
                '?x1 ?x0 ' + value + '. '
                                    """
        FILTER regex(?x0, "http://rdf.freebase.com/ns/")
        }
        }
        """)
        # print(query1)

        self.sparql.setQuery(query1)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query1)
            exit(0)
        for result in results['results']['bindings']:
            in_attributes.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

        return in_attributes



    def get_in_relations(self, entity: str):
        in_relations = set()

        query1 = ("""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/> 
                SELECT (?x0 AS ?value) WHERE {
                SELECT DISTINCT ?x0  WHERE {
                """
                '?x1 ?x0 ' + ':' + entity + '. '
                                            """
        FILTER regex(?x0, "http://rdf.freebase.com/ns/")
        }
        }
        """)
        # print(query1)

        self.sparql.setQuery(query1)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query1)
            exit(0)
        for result in results['results']['bindings']:
            in_relations.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

        return in_relations


    def get_in_entities(self, entity: str, relation: str):
        neighbors = set()

        query1 = ("""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/> 
                SELECT (?x1 AS ?value) WHERE {
                SELECT DISTINCT ?x1  WHERE {
                """
                '?x1' + ' :' + relation + ' :' + entity + '. '
                                                            """
                    FILTER regex(?x1, "http://rdf.freebase.com/ns/")
                    }
                    }
                    """)
        # print(query1)

        self.sparql.setQuery(query1)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query1)
            exit(0)
        for result in results['results']['bindings']:
            neighbors.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

        return neighbors



    def get_out_relations(self, entity: str):
        out_relations = set()

        query2 = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT (?x0 AS ?value) WHERE {
            SELECT DISTINCT ?x0  WHERE {
            """
                ':' + entity + ' ?x0 ?x1 . '
                                """
        FILTER regex(?x0, "http://rdf.freebase.com/ns/")
        }
        }
        """)
        # print(query2)

        self.sparql.setQuery(query2)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query2)
            exit(0)
        for result in results['results']['bindings']:
            out_relations.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

        return out_relations


    def get_out_entities(self, entity: str, relation: str):
        neighbors = set()

        query2 = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT (?x1 AS ?value) WHERE {
            SELECT DISTINCT ?x1  WHERE {
            """
                ':' + entity + ' :' + relation + ' ?x1 . '
                                                """
                        FILTER regex(?x1, "http://rdf.freebase.com/ns/")
                        }
                        }
                        """)
        # print(query2)

        self.sparql.setQuery(query2)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query2)
            exit(0)
        for result in results['results']['bindings']:
            neighbors.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

        return neighbors





if __name__ == '__main__':
    sparql_executer = SparqlExecuter()
    query = """
    PREFIX ns: <http://rdf.freebase.com/ns/>                                                                                                                                                                    
SELECT DISTINCT ?x                                                                                                                                                                                          
WHERE {                                                                                                                                                                                                     
FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))                                                                                                                                                                                                                                                                                       
ns:m.025tbxf ns:type.object.type ?x .                                                                                                                                                                                                                                                                                                                                         
}     
    """
    results = sparql_executer.execute_query(query)
    print(results)