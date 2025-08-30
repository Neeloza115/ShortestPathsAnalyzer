from Part6 import WeightedGraph
import heapq

#Item class for items on the heap
class Item:
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __lt__(self, next):
        return self.value < next.value

#Function to make a path between two nodes given a predecessors dictionary
def make_path(src: int, dst: int, predecessors: dict[int, int]) -> None | list[int]:
    current = dst
    path = [ dst ]

    while predecessors.get(current) != None:
        current = predecessors[current]
        path.append(current)

        if current == src:
            break


    if path[-1] != src:
        return None

    return list(reversed(path))

#Tries to reduce the value at a given key in the heap
def reduce_key(heap: list[Item], new_item: Item) -> None:
    for i in range(len(heap)):
        if heap[i].key == new_item.key:
            if new_item.value < heap[i].value:
                heap[i] = new_item
                heapq.heapify(heap)

            return
        
    #Element is not in the heap, so add it
    heapq.heappush(heap, new_item)

#A* Function for Part 4 and Part 5
#Algorithm modified from Lecture 18 Page 8
def A_star(graph: WeightedGraph, src: int, dst: int, heuristic: dict[int, float]):
    #Initializations
    #Not a normal list, used as a heap with the heapq library
    #Stored as key value pairs (value, key) because heapq likely sorts based on first element
    open_set_heap = []  # We will push items to the heap, so we don't need to preinitialize all nodes
    heapq.heappush(open_set_heap, Item(src, heuristic[src]))  # Push the source into the heap

    predecessors = {}

    gScore = {}

    for node in graph.get_nodes():
        gScore[node] = float("inf")

    gScore[src] = 0

    while len(open_set_heap) > 0:
        current = heapq.heappop(open_set_heap).key

        if current == dst:
            break

        for next in graph.get_adj_nodes(current):
            newScore = gScore[current] + graph.w(current, next)

            if newScore < gScore[next]:
                predecessors[next] = current
                gScore[next] = newScore

                reduce_key(open_set_heap, Item(next, newScore + heuristic[next]))

    return (predecessors, make_path(src, dst, predecessors))