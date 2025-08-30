from Part6 import WeightedGraph
from Part2 import dijkstra, bellman_ford


def floyd_warshall(graph: WeightedGraph):
    # Create 2 adjacency matrices
    V = len(graph.nodes)
    #This returns the shortest path between vertex i and j
    matrix = [[float('inf') for _ in range(V)] for _ in range(V)]
    #This returns the second last vertex on the shortest path between i and j
    prev = [[-1] * V for _ in range(V)]

    # Set distance to itself as 0
    for u in range(V):
        matrix[u][u] = 0

    # Populate matrix with the weights of the edges and prev with the source node
    for start, end in graph.edges:
        weight = graph.w(start, end)
        matrix[start][end] = weight
        prev[start][end] = start

    # for each edge pair, populate the table with the path with the smallest weight and prev with the second-last vertex of the shortest path
    for k in range(V):
        for i in range(V):
            for j in range(V):
                if matrix[i][j] > matrix[i][k] + matrix[k][j]:
                    matrix[i][j] = matrix[i][k] + matrix[k][j]
                    prev[i][j] = prev[k][j]

    return "Shortest Paths:", matrix, "Second Last Vertex:", prev

def all_pairs_dijkstra(graph: WeightedGraph):
    # This will return the start, end, distance, path and second-last-vertex
    all_paths = {}

    # Run Dijkstra from part 2 for all source nodes
    for source in graph.get_nodes():
        result_from_source = dijkstra(graph, source, k=graph.get_num_of_nodes() - 1)
        all_paths[source] = {}

        for target in result_from_source:
            distance, path = result_from_source[target]
            # Get second-to-last vertex if it exists
            second_last = path[-2] if len(path) >= 2 else None
            # Store the distance, path, and second-to-last vertex
            all_paths[source][target] = {
                "distance": distance,
                "path": path,
                "second_last": second_last
            }

    return all_paths

def all_pairs_bellman_ford(graph: WeightedGraph):
    # This will return the start, end, distance, path and second-last-vertex
    all_paths = {}

    # Run Bellman-Ford from part 2 for all source nodes
    for source in graph.get_nodes():
        result_from_source = bellman_ford(graph, source, k=graph.get_num_of_nodes() - 1)
        all_paths[source] = {}

        for target in result_from_source:
            distance, path = result_from_source[target]
            # Get second-to-last vertex if it exists
            second_last = path[-2] if len(path) >= 2 else None
            # Store the distance, path, and second-to-last vertex
            all_paths[source][target] = {
                "distance": distance,
                "path": path,
                "second_last": second_last
            }

    return all_paths


if __name__ == '__main__':
    # Creates a weighted graph and adds weighted edges
    g = WeightedGraph()
    g.add_edge(0, 1, 2)
    g.add_edge(0, 2, 5)
    g.add_edge(1, 2, 1)
    g.add_edge(2, 3, 2)
    g.add_edge(3, 0, 4)

    res = all_pairs_dijkstra(g)

    # Display the results for Dijkstra
    print("All pairs Dijkstra's")
    for src in res:
        for dst in res[src]:
            info = res[src][dst]
            print("Start: ", src, "End: ", dst, "Path: ", info['path'], "Distance :", info['distance'], "Second last vertex:",  info['second_last'])

    res2 = all_pairs_bellman_ford(g)

    # Display the results for Bellman_ford
    print("All pairs Bellman-Ford")
    for src in res:
        for dst in res[src]:
            info = res[src][dst]
            print("Start: ", src, "End: ", dst, "Path: ", info['path'], "Distance :", info['distance'],
                  "Second last vertex:", info['second_last'])

    print("Floyd-Warshall")
    #Display the results for Floyd Warshall
    res3 = floyd_warshall(g)
    print(res3)
