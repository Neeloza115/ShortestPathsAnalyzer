#! Assuming that we are to use graph implementations from part 6
from Part6 import WeightedGraph
import copy
import random
import time
import timeit 
import matplotlib.pyplot as plt
import numpy as np
# Assuming its okay to import theses since in part 4 we can import
import heapq

#* 2.1
# k how many times an edge can be relaxed
# k must be in the range (0,N-1) where N is the number of nodes in the graph
def dijkstra(graph: WeightedGraph, source: int, k: int):
    #Tries to reduce the value at a given key in the heap or makes it if none existent
    def reduce_key(heap: list[int], distance: int, node: float) -> None:
        for i in range(len(heap)):
            if heap[i][1] == node:
                if distance < heap[i][0]:
                    heap[i] = (distance, node)
                    heapq.heapify(heap)
                return
        
        heapq.heappush(heap, (distance, node))
    
    # clamping k into the range
    k = max(0, min(k, graph.get_num_of_nodes() - 1))

    # path dictionary which maps a node to the distance and the path
    # i.e. key is int and value is tuple (int, list[int])
    rtn = {}
    pq = []
    relaxations = {}
    for node in graph.get_nodes():
        relaxations[node] = k
    
    # initialize the path dictionary
    for node in graph.nodes:
        rtn[node] = (float('inf'), [])

    # set the distance of the source node to 0
    rtn[source] = (0, [source])
    heapq.heappush(pq, (0, source))

    while len(pq) > 0:
        min_dist_node = heapq.heappop(pq)[1]
        
        #* relaxing node min_dist_node
        for adj_node in graph.get_adj_nodes(min_dist_node):

            alternate_distance = rtn[min_dist_node][0] + graph.w(min_dist_node, adj_node)
            
            if alternate_distance < rtn[adj_node][0]:
                # making sure we are only realxing the node k times
                if relaxations[adj_node] <= 0:
                    continue
                
                # removing one relaxtion from the total possible on this node
                relaxations[adj_node] -= 1

                rtn[adj_node] = (alternate_distance, copy.copy(rtn[min_dist_node][1]) + [adj_node])
                reduce_key(pq, alternate_distance, adj_node)

    return rtn

#* 2.2
# k how many times an edge can be relaxed
# k must be in the range (0,N-1) where N is the number of nodes in the graph
def bellman_ford(graph: WeightedGraph, source: int, k: int):
    # clamping k into the range
    k = max(0, min(k, graph.get_num_of_nodes() - 1))
    
    # path dictionary which maps a node to the distance and the path
    # i.e. key is int and value is tuple (int, list[int])
    rtn = {}
    q = []
    onQ = [False for _ in range(0, graph.get_num_of_nodes())]
    relaxations = {}
    for node in graph.get_nodes():
        relaxations[node] = k
    
    # initialize the path dictionary
    for node in graph.nodes:
        rtn[node] = (float('inf'), [])
        
    # set the distance of the source node to 0
    rtn[source] = (0, [source])
    q.append(source)
    onQ[source] = True
    
    # this does not work with negative cycles
    while len(q) > 0:
        # pop the node from the queue
        current_node = q.pop(0)
        onQ[current_node] = False
        
        #* relaxing node current_node
        for adj_node in graph.get_adj_nodes(current_node):
            alternate_distance = rtn[current_node][0] + graph.w(current_node, adj_node)
            if alternate_distance < rtn[adj_node][0]:
                # making sure we are only relaxing the node k times
                if relaxations[adj_node] == 0:
                    continue
                
                # removing one relaxation from the total possible on this node
                relaxations[adj_node] -= 1

                rtn[adj_node] = (alternate_distance, copy.copy(rtn[current_node][1]) + [adj_node])
                if not onQ[adj_node]:
                    q.append(adj_node)
                    onQ[adj_node] = True

    return rtn

# function to generate random graphs 
def create_random_graph(nodes, edges, weightMin = 1, weightMax = 100) -> WeightedGraph:
    if edges > nodes * nodes: # making sure the number of edges is not too large
        return None

    graph = WeightedGraph()
    for i in range(0, nodes):
        graph.add_node(i)

    i = 0 
    while i < edges:
        src = random.randint(0, nodes - 1) # getting a random node
        dst = random.randint(0, nodes - 1) # getting a random node
        weight = random.randint(weightMin, weightMax) # getting a random weight
        if dst not in graph.get_adj_nodes(src):
            graph.add_edge(src, dst, weight)
        else:
            i -= 1 # we want to try and add another edge since the current one already existed therefore we did not add one

        i += 1

    return graph

#* if xValues is None then defaults to 0 to len(arr) while iterating by one
#* if meanTitles is None then the mean lines will have no labels
#* if means is None then no mean lines will be drawn
def draw_plot(arr, meanTitles, means, title, xLabel = "Not Given!!", yLabel = "Not Given!!", save=False, xValues = None, xValuesRotation = 0):
    if xValues == None:
        xValues = [i for i in range(1, len(arr) + 1)]
    x = np.arange(0, len(xValues),1)
    fig=plt.figure(figsize=(20,8))
    plt.bar(xValues, height=arr) # added this for bars
    plt.tick_params(axis='x', rotation=xValuesRotation) # added this to rotate the x axis labels
    if (means != None):
        for i in range(0,len(means)):
            if meanTitles != None:
                plt.axhline(means[i],color="red",linestyle="--",label=meanTitles[i] + " = " + str(means[i]))
            else:
                plt.axhline(means[i],color="red",linestyle="--")

        plt.legend()

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title) # added a input to change the title
    if save: # added this to check if we should save the plot or draw it
        plt.savefig(title)
        plt.close()
    else:
        plt.show()

def test_dijkstra_simple_graph():
    # Create a simple graph
    graph = WeightedGraph()
    graph.add_edge(0, 1, 4)
    graph.add_edge(0, 2, 1)
    graph.add_edge(2, 1, 2)
    graph.add_edge(1, 3, 1)
    graph.add_edge(2, 3, 5)

    result = dijkstra(graph, source=0, k=graph.get_num_of_nodes() - 1)

    # Expected results
    expected = {
        0: (0, [0]),
        1: (3, [0, 2, 1]),
        2: (1, [0, 2]),
        3: (4, [0, 2, 1, 3])
    }

    assert result == expected, f"Expected {expected}, but got {result}"

def test_dijkstra_no_relaxation():
    # Create a simple graph
    graph = WeightedGraph()
    graph.add_edge(0, 1, 4)
    graph.add_edge(0, 2, 1)
    graph.add_edge(2, 1, 2)
    graph.add_edge(1, 3, 1)
    graph.add_edge(2, 3, 5)

    # Test dijkstra with k = 0 (no relaxation)
    result = dijkstra(graph, source=0, k=0)

    # Expected results (no relaxation, all distances except source are infinity)
    expected = {
        0: (0, [0]),
        1: (float('inf'), []),
        2: (float('inf'), []),
        3: (float('inf'), [])
    }

    assert result == expected, f"Expected {expected}, but got {result}"

def test_bellman_ford_simple_graph():
    # Create a simple graph
    graph = WeightedGraph()
    graph.add_edge(0, 1, 4)
    graph.add_edge(0, 2, 1)
    graph.add_edge(2, 1, 2)
    graph.add_edge(1, 3, 1)
    graph.add_edge(2, 3, 5)

    result = bellman_ford(graph, source=0, k=graph.get_num_of_nodes() - 1)

    # Expected results
    expected = {
        0: (0, [0]),
        1: (3, [0, 2, 1]),
        2: (1, [0, 2]),
        3: (4, [0, 2, 1, 3])
    }

    assert result == expected, f"Expected {expected}, but got {result}"

def test_bellman_ford_no_relaxation():
    # Create a simple graph
    graph = WeightedGraph()
    graph.add_edge(0, 1, 4)
    graph.add_edge(0, 2, 1)
    graph.add_edge(2, 1, 2)
    graph.add_edge(1, 3, 1)
    graph.add_edge(2, 3, 5)

    # Test bellman_ford with k = 0 (no relaxation)
    result = bellman_ford(graph, source=0, k=0)

    # Expected results (no relaxation, all distances except source are infinity)
    expected = {
        0: (0, [0]),
        1: (float('inf'), []),
        2: (float('inf'), []),
        3: (float('inf'), [])
    }

    assert result == expected, f"Expected {expected}, but got {result}"

# test_bellman_ford_simple_graph()
# test_bellman_ford_no_relaxation()
    
# test_dijkstra_simple_graph()
# test_dijkstra_no_relaxation()
# print("Finished")

#* 2.3
def experiment():
    num_trials = 100
    num_graphs_in_trial = 100
    # the different combinations of nodes and edges to test
    N = 50
    nodes_arr = [N,    N,       N,       N,   N]
    edges_arr = [N**2, N*(N-1), int(N*(N/2)), N*2, N]
    averages = [[],[]]
    
    total_start = timeit.default_timer()
    for test_type in range(0, len(nodes_arr)):
        nodes = nodes_arr[test_type]
        edges = edges_arr[test_type]
        k = nodes - 1 # so that we compare the full functionality of the two algorithms
        
        print("Testing with " + str(nodes) + " nodes and " + str(edges) + " edges")
        
        dijkstra_times = []
        bellman_times = []
        for i in range(0, num_trials):
            print("Trial " + str(i+1) + " of " + str(num_trials))

            graphs = [create_random_graph(nodes, edges) for _ in range(num_graphs_in_trial)]
            dijkstra_average = 0
            bellman_average = 0
            
            for graph in graphs:
                start = timeit.default_timer()
                dijkstra(graph, 0, k) # this does not edit anything in graph so there is no need to copy
                end = timeit.default_timer()
                dijkstra_average += (end - start) * 1000

                start = timeit.default_timer()
                bellman_ford(graph, 0, k)
                end = timeit.default_timer()
                bellman_average += (end - start) * 1000

            dijkstra_times.append(dijkstra_average / num_graphs_in_trial)
            bellman_times.append(bellman_average / num_graphs_in_trial)

        averages[0].append(sum(dijkstra_times)/num_trials)
        averages[1].append(sum(bellman_times)/num_trials)
        draw_plot(dijkstra_times, ["Dijkstras Average"], 
                [sum(dijkstra_times)/num_trials], "Dijkstras Average times for " + str(edges) + " edges and " + str(nodes) + " nodes", 
                "Trial Number", "Time (ms)", True)
        draw_plot(bellman_times, ["Bellman-Fords Average"], 
                [sum(bellman_times)/num_trials], "Bellman-Fords Average times for " + str(edges) + " edges and " + str(nodes) + " nodes", 
                "Trial Number", "Time (ms)", True)
    
    print("Averages for Dijkstra: " + str(averages[0]))
    print("Averages for Bellman-Ford: " + str(averages[1]))
    total_end = timeit.default_timer()
    print("Total time for all trials: " + str(total_end - total_start) + " seconds")

if __name__ == "__main__":
    experiment()