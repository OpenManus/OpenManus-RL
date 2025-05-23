from typing import Iterable, List, Dict, Union, Tuple, Optional

from pydantic import BaseModel


class Graph:
    def __init__(self, node_count: int, edges: Dict[Tuple[int, int], int]):
        """
        edges: {(source, target): edge_weight}}
        """
        self.node_count = node_count
        self.edges = edges

    def iterate_edges(self) -> Iterable[Tuple[int, int, int]]:
        for (source, target), weight in self.edges.items():
            yield source, target, weight


class Edge(BaseModel):
    from_node: int
    to_node: int
    capacity: int
    flow: int = 0


class MaxFlow:
    def __init__(self, graph: Graph, src: int, dst: int) -> None:
        assert (
            graph.node_count > src >= 0
        ), "src node out of range, expected [0, {}), got {}".format(
            graph.node_count, src
        )
        assert (
            graph.node_count > dst >= 0
        ), "dst node out of range, expected [0, {}), got {}".format(
            graph.node_count, dst
        )

        self.src = src
        self.dst = dst
        self.graph = graph
        self.adjacent_edges: List[List[Edge]] = [[] for _ in range(graph.node_count)]
        self.edges_dict: Dict[Tuple[int, int], Edge] = {}

        for source, target, weight in self.graph.iterate_edges():
            if (source, target) in self.edges_dict:
                self.edges_dict[(source, target)].capacity += weight
            else:
                self.edges_dict[(source, target)] = Edge(
                    from_node=source, to_node=target, capacity=weight
                )
                self.edges_dict[(target, source)] = Edge(
                    from_node=target, to_node=source, capacity=0
                )
                self.adjacent_edges[source].append(self.edges_dict[(source, target)])
                self.adjacent_edges[target].append(self.edges_dict[(target, source)])

        self.max_flow = self.compute_max_flow()

    def compute_max_flow(self) -> int:
        max_flow = 0
        while True:
            augmenting_path = self.find_augmenting_path()
            if not augmenting_path:
                break
            bottleneck = min([edge.capacity - edge.flow for edge in augmenting_path])
            for edge in augmenting_path:
                edge.flow += bottleneck
                self.edges_dict[(edge.to_node, edge.from_node)].flow -= bottleneck
            max_flow += bottleneck
        return max_flow

    def find_augmenting_path(self) -> Optional[List[Edge]]:
        # BFS
        visited = [False] * self.graph.node_count
        visited[self.src] = True
        queue = [self.src]
        prev: List[Union[None, Edge]] = [None] * self.graph.node_count
        while queue:
            node = queue.pop(0)
            for edge in self.adjacent_edges[node]:
                if not visited[edge.to_node] and edge.capacity > edge.flow:
                    visited[edge.to_node] = True
                    prev[edge.to_node] = edge
                    queue.append(edge.to_node)
                    if edge.to_node == self.dst:
                        break
        if not visited[self.dst]:
            return None
        flow_path = []
        node = self.dst
        while prev[node]:
            flow_path.append(prev[node])
            node = prev[node].from_node
        flow_path.reverse()
        return flow_path


if __name__ == "__main__":
    g = Graph(
        node_count=8,
        edges={
            (0, 3): 100,
            (3, 2): 60,
            (3, 4): 10,
            (3, 5): 20,
            (3, 6): 8,
            (3, 7): 0,
            (2, 1): 50,
            (4, 1): 30,
            (5, 1): 20,
            (6, 1): 20,
            (7, 1): 20,
        },
    )
    print(g)
    for edge in g.iterate_edges():
        print(edge)
    m = MaxFlow(g, 0, 1)
    print(m.edges_dict)
