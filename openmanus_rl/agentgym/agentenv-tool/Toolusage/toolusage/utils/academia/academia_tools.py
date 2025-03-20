import pandas as pd
import jsonlines
import json
import re
from dotenv import load_dotenv
import pickle
import os
import copy

def log_path(func):
    def wrapper(*args, **kwargs):
        if "action_path" in kwargs.keys():
            action_path = kwargs["action_path"]
            kwargs.pop("action_path")
            success, result = func(*args, **kwargs)

            # convert value in kwargs to string
            # for key, value in kwargs.items():
                # kwargs[key] = str(value)

            if success:
                action_path.append({
                    "Action" : func.__name__,
                    "Action Input" : str(kwargs),
                    "Observation": result,
                    "Subgoal": result,
                })
                return result
            else:
                action_path.append({
                    "Action" : func.__name__,
                    "Action Input" : str(kwargs),
                    "Observation": result,
                    "Subgoal": "Calling " + func.__name__ + " with " + str(kwargs) + " failed",
                })
                return result
        else:
            return func(*args, **kwargs)
    return wrapper

class academia_toolkits:
    # init
    def __init__(self, path, dataset):
        self.paper_net = None
        self.author_net = None
        self.id2title_dict = None
        self.title2id_dict = None
        self.id2author_dict = None
        self.author2id_dict = None
        self.path = path
        self.dataset = dataset

    def load_graph(self, graph_name):
        # print(graph_name)       
        if graph_name == 'dblp' or graph_name == "DBLP":
            with open('{}/data/tool-query/academia/raw/paper_net.pkl'.format(self.path), 'rb') as f:
                self.paper_net = pickle.load(f)

            with open('{}/data/tool-query/academia/raw/author_net.pkl'.format(self.path), 'rb') as f:
                self.author_net = pickle.load(f)
            
            with open("{}/data/tool-query/academia/raw/title2id_dict.pkl".format(self.path), "rb") as f:
                self.title2id_dict = pickle.load(f)
            with open("{}/data/tool-query/academia/raw/author2id_dict.pkl".format(self.path), "rb") as f:
                self.author2id_dict = pickle.load(f)
            with open("{}/data/tool-query/academia/raw/id2title_dict.pkl".format(self.path), "rb") as f:
                self.id2title_dict = pickle.load(f)
            with open("{}/data/tool-query/academia/raw/id2author_dict.pkl".format(self.path), "rb") as f:
                self.id2author_dict = pickle.load(f)
            return True, "DBLP data is loaded, including two sub-graphs: AuthorNet and PaperNet."
        else:
            return False, "{} is not a valid graph name.".format(graph_name)
    
    @log_path
    def loadPaperNet(self):
        with open('{}/data/tool-query/academia/raw/paper_net.pkl'.format(self.path), 'rb') as f:
            self.paper_net = pickle.load(f)

        
        with open("{}/data/tool-query/academia/raw/title2id_dict.pkl".format(self.path), "rb") as f:
            self.title2id_dict = pickle.load(f)
        with open("{}/data/tool-query/academia/raw/id2title_dict.pkl".format(self.path), "rb") as f:
            self.id2title_dict = pickle.load(f)
        with open("{}/data/tool-query/academia/raw/author2id_dict.pkl".format(self.path), "rb") as f:
            self.author2id_dict = pickle.load(f)
        with open("{}/data/tool-query/academia/raw/id2author_dict.pkl".format(self.path), "rb") as f:
            self.id2author_dict = pickle.load(f)

        return True, "PaperNet is loaded."

    @log_path
    def loadAuthorNet(self):
        with open('{}/data/tool-query/academia/raw/author_net.pkl'.format(self.path), 'rb') as f:
            self.author_net = pickle.load(f)

        with open("{}/data/tool-query/academia/raw/title2id_dict.pkl".format(self.path), "rb") as f:
            self.title2id_dict = pickle.load(f)
        with open("{}/data/tool-query/academia/raw/id2title_dict.pkl".format(self.path), "rb") as f:
            self.id2title_dict = pickle.load(f)
        with open("{}/data/tool-query/academia/raw/author2id_dict.pkl".format(self.path), "rb") as f:
            self.author2id_dict = pickle.load(f)
        with open("{}/data/tool-query/academia/raw/id2author_dict.pkl".format(self.path), "rb") as f:
            self.id2author_dict = pickle.load(f)

        return True, "AuthorNet is loaded."

    @log_path
    def neighbourCheck(self, graph, node):
        if graph == "PaperNet" and self.paper_net == None:
            return False, "Please load the PaperNet first."
        elif graph == "AuthorNet" and self.author_net == None:
            return False, "Please load the AuthorNet first."
        try:
            if graph == 'PaperNet':
                graph = self.paper_net
                dictionary = self.title2id_dict
                inv_dict = self.id2title_dict
                
                if node not in dictionary.keys():
                    return False, "There is no node named {} in the PaperNet.".format(node)

            elif graph == 'AuthorNet':
                graph = self.author_net
                dictionary = self.author2id_dict
                inv_dict = self.id2author_dict

                if node not in dictionary.keys():
                    return False, "There is no node named {} in the AuthorNet.".format(node)

            neighbour_list = []
            for neighbour in graph.neighbors(dictionary[node]):
                neighbour_list.append(inv_dict[neighbour])
            return True, neighbour_list
        except Exception as e:
            return False, type(e).__name__ + "(" + str(e) + ")"  

    @log_path
    def paperNodeCheck(self, node=None):
        if self.paper_net == None:
            return False, "Please load the PaperNet first."
        try:
            graph = self.paper_net
            dictionary = self.title2id_dict
            inv_dict = self.id2title_dict

            if node not in dictionary.keys():
                return False, "There is no node named {} in the PaperNet.".format(node)

            return True, graph.nodes[dictionary[node]]
        except Exception as e:
            return False, type(e).__name__ + "(" + str(e) + ")"  
    
    @log_path
    def authorNodeCheck(self, node=None):
        if self.author_net == None:
            return False, "Please load the AuthorNet first."
        try:

            graph = self.author_net
            dictionary = self.author2id_dict
            inv_dict = self.id2author_dict

            if node not in dictionary.keys():
                return False, "There is no node named {} in the AuthorNet.".format(node)

            author_node_info = copy.deepcopy( graph.nodes[dictionary[node]] )
            # for idx, paper in enumerate(author_node_info['papers']):
                # author_node_info['papers'][idx] = self.id2title_dict[paper]
            return True, author_node_info         

        except Exception as e:
            return False, type(e).__name__ + "(" + str(e) + ")"  

    def check_nodes(self, graph, node):
        if self.paper_net == None:
            return False, "Please load the graph first."
        try:
            if graph == 'PaperNet':
                graph = self.paper_net
                dictionary = self.title2id_dict
                inv_dict = self.id2title_dict
                return True, graph.nodes[dictionary[node]]
            elif graph == 'AuthorNet':
                graph = self.author_net
                dictionary = self.author2id_dict
                inv_dict = self.id2author_dict

                author_node_info = copy.deepcopy( graph.nodes[dictionary[node]] )
                for idx, paper in enumerate(author_node_info['papers']):
                    author_node_info['papers'][idx] = self.id2title_dict[paper]
                return True, author_node_info
        except Exception as e:
            return False, type(e).__name__ + "(" + str(e) + ")"  

    @log_path
    def authorEdgeCheck(self, node1=None, node2=None):
        if self.author_net == None:
            return False, "Please load the AuthorNet first."
        try:
            graph = self.author_net
            dictionary = self.author2id_dict
            inv_dict = self.id2title_dict

            if node1 not in dictionary.keys():
                return False, "There is no node named {} in the AuthorNet.".format(node1)

            if node2 not in dictionary.keys():
                return False, "There is no node named {} in the AuthorNet.".format(node2)

            if dictionary[node2] not in graph.neighbors(dictionary[node1]):
                return False, "There is no edge between {} and {}.".format(node1, node2)

            edge = graph.edges[dictionary[node1], dictionary[node2]]
            new_edge = copy.deepcopy(edge)
            # print(edge)
            for id in range(len(edge['collaborative_papers'])):
                new_edge['collaborative_papers'][id] = inv_dict[edge['collaborative_papers'][id]]
            return True, new_edge
        except Exception as e:
            return False, type(e).__name__ + "(" + str(e) + ")"        
    
    @log_path
    def paperEdgeCheck(self, node1=None, node2=None):
        if self.paper_net == None:
            return False, "Please load the PaperNet first."
        try:
            graph = self.paper_net
            dictionary = self.title2id_dict
            inv_dict = self.id2title_dict

            if node1 not in dictionary.keys():
                return False, "There is no node named {} in the PaperNet.".format(node1)
            
            if node2 not in dictionary.keys():
                return False, "There is no node named {} in the PaperNet.".format(node2)

            if dictionary[node2] not in graph.neighbors(dictionary[node1]):
                return False, "There is no edge between {} and {}.".format(node1, node2)

            edge = graph.edges[dictionary[node1], dictionary[node2]]
            return True, edge
        except Exception as e:
            return False, type(e).__name__ + "(" + str(e) + ")"  

    # check the attributes of the edges
    def check_edges(self, graph, node1, node2):
        if self.paper_net == None:
            return False, "Please load the graph first."
        try:
            
            if graph == 'PaperNet':
                graph = self.paper_net
                dictionary = self.title2id_dict
                inv_dict = self.id2title_dict
                edge = graph.edges[dictionary[node1], dictionary[node2]]
                return True, edge
            elif graph == 'AuthorNet':
                graph = self.author_net
                dictionary = self.author2id_dict
                inv_dict = self.id2title_dict
                edge = graph.edges[dictionary[node1], dictionary[node2]]
                new_edge = copy.deepcopy(edge)
                # print(edge)
                for id in range(len(edge['papers'])):
                    new_edge['papers'][id] = inv_dict[edge['papers'][id]]
                return True, new_edge

        except Exception as e:
            return False, type(e).__name__ + "(" + str(e) + ")"  
        
    @log_path
    def finish(self, answer):
        if type(answer) == list:
            answer = sorted(answer)
        return True, answer

if __name__ == "__main__":
    load_dotenv()
    academia_toolkits = academia_toolkits(path=os.environ["PROJECT_PATH"])
    logs = academia_toolkits.load_graph('dblp')
    print( str(academia_toolkits.check_neighbours("AuthorNet", "Mucong Li")) )
    print( str(academia_toolkits.check_edges("AuthorNet", "Chao Zhang", "Weihong Lin")) )