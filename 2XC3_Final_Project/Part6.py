
# this is directed
#! assuming this is not weighted
class Graph:
    def __init__(self):
        # dict of nodes such that its a list of adjacent nodes
        self.nodes = {}

    def get_adj_list(self):
        # Return the adjacency list
        return self.nodes

    def get_adj_nodes(self, node: int) -> list[int]:
        if node not in self.nodes:
            return []
        else:
            return self.nodes[node]
    
    def add_node(self, node: int) -> None:
        if node not in self.nodes:
            self.nodes[node] = []

    def get_nodes(self) -> list[int]:
        return list(self.nodes.keys())
    
    # if the node does not exists it will be created
    # def add_edge(self, start: int, end: int, w: float) -> None:
    #! since i assumed that the graph is not weighted, i will not include the w input value in here and reimplement in the weighted graph implementation
    def add_edge(self, start: int, end: int) -> None:
        if start not in self.nodes:
            self.nodes[start] = []
        if end not in self.nodes:
            self.nodes[end] = []
        self.nodes[start].append(end)
        # self.edges[(start, end)] = w

    def get_num_of_nodes(self) -> int:
        return len(self.nodes)
    
    def w(node: int) -> float:
        return float(node)

#! assuming this needs the add_edge method to be reimplemented since the original should not have weight
class WeightedGraph(Graph):
    def __init__(self):
        super().__init__()
        # dict such that key is (start_node, end_node) and value is the weight
        self.edges = {}
    
    # if the node does not exists it will be created
    def add_edge(self, start: int, end: int, w: float) -> None:
        super().add_edge(start, end)
        self.edges[(start, end)] = w

    # returns the weight of a given edge
    def w(self, start: int, end: int) -> float:
        if self.edges.get((start, end)) == None:
            return float("Infinity")
        
        return self.edges[(start, end)]

class HeuristicGraph(WeightedGraph):
    def __init__(self):
        super().__init__()
        # dict such that key is the node and value is the heuristic value
        self.__heuristics = {}
    
    # if the node does not exists it will NOT be created
    def add_edge(self, start: int, end: int, w: float) -> None:
        if start not in self.nodes or end not in self.nodes:
            return
        
        self.nodes[start].append(end)
        self.edges[(start, end)] = w

    #! overloading this function since each node must have a heuristic value on creation
    def add_node(self, node: int, h: float) -> None:
        if node not in self.nodes:
            self.nodes[node] = []
        self.__heuristics[node] = h

    def get_heuristic(self) -> dict[int, float]:
        return self.__heuristics
    
from Part2 import dijkstra
from Part2 import bellman_ford
from Part4 import A_star
class SPAlgorithm:
    def __init__(self):
        pass

    def calc_sp(self, graph: Graph, source: int, dest: int) -> None | float:
        return None
    
class Dijkstra(SPAlgorithm):
    def __init__(self):
        pass

    def calc_sp(self, graph: Graph, source: int, dest: int) -> float:
        return dijkstra(graph, source, graph.get_num_of_nodes()-1)[dest][0]
    
class Bellman_Ford(SPAlgorithm):
    def __init__(self):
        pass

    def calc_sp(self, graph: Graph, source: int, dest: int) -> float:
        return bellman_ford(graph, source, graph.get_num_of_nodes()-1)[dest][0]
    
class A_Star(SPAlgorithm):
    def __init__(self):
        pass

    def calc_sp(self, graph: Graph, source: int, dest: int) -> float:
        path = A_star(graph, source, dest, graph.get_num_of_nodes()-1)[1]
        distance = 0
        current = path[0]
        for i in range(1,len(path)):
            distance += graph.w(current, path[i])
            current = path[i]
        return distance

class ShortPathFinder:
    def __init__(self):
        self.graph = None
        self.algorithm = None

    def calc_short_path(self, source: int, dest: int) -> None | float:
        if self.algorithm == None:
            return None
        
        return self.algorithm.calc_sp(self.graph, source, dest)

    def set_graph(self, graph: Graph):
        self.graph = graph

    def set_algorithm(self, algorithm: SPAlgorithm):
        self.algorithm = algorithm