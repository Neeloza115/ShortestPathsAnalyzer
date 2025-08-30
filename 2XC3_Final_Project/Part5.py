from Part6 import WeightedGraph
from Part4 import A_star
from Part2 import dijkstra, draw_plot
import math
#import time
import timeit
import heapq
from collections import defaultdict
import copy
import random

class Station:
    def __init__(self,data_arr):
        if len(data_arr) != 8:
            raise Exception(f"Incorrect data format on station creation, provided data ${data_arr}")

        #Assigns each attribute from the associated value in the array
        self.id = int(data_arr[0])
        self.latitude = float(data_arr[1])
        self.longitude = float(data_arr[2])
        self.name = data_arr[3]
        self.display_name = data_arr[4]
        self.zone = float(data_arr[5])
        self.total_lines = int(data_arr[6])
        self.rail = int(data_arr[7])

    #Returns the euclidean distance between this station and another
    def distance_to(self, station):
        return math.sqrt((self.longitude - station.longitude) ** 2 + (self.latitude - station.latitude) ** 2)

    def __str__(self):
        return f"Id: {self.id}, Lat: {self.latitude}, Long: {self.longitude}, Name: {self.name}, Display: {self.display_name}, Zone: {self.zone}, Total: {self.total_lines}, Rail: {self.rail}"

#Custom function because the values can contain commas
def parse_csv_line(line: str) -> list[str]:
    if line == "":
        return []

    result = [ "" ]

    in_quotes = False

    #Splits the line based on commas that are not surrounded by quotes
    for c in line:
        if c == ',' and not in_quotes:
            result.append("")
        elif c == '"':
            in_quotes = not in_quotes
        else:
            result[-1] += c

    return result

#Reads london_stations.csv, returns a weighted graph containing the stations and a dictionary matching each station to its id
def read_stations_csv() -> tuple[WeightedGraph, dict[int, Station]]:
    file = open('london_stations.csv', 'r')

    graph = WeightedGraph()

    stations = {}

    #Skip header line
    file.readline()

    line = file.readline()

    #Loop until the end of the file
    while line != "":
        items = parse_csv_line(line)

        #Convert the id to int separately
        id = int(items[0])

        #Add the node based on the id
        graph.add_node(id)

        #Builds the station based on the line of info
        stations[id] = Station(items)

        line = file.readline()

    #Close the file stream to prevent resource leak
    file.close()

    return (graph, stations)

#Reads london_connections.csv and adds the connects to the graph based on it and dictionaries storing each line and the lines each edge has
def read_connections_csv(graph: WeightedGraph, stations: dict[int, Station]) -> tuple[WeightedGraph, dict[int, set[int], dict[tuple[int, int], set[int]]]]:
    file = open('london_connections.csv', 'r')

    lines = {}

    edge_lines = {}

    #Skip header line
    file.readline()

    line = file.readline()

    #Loop until the end of the file
    while line != "":
        items = parse_csv_line(line)

        #Reads parameters from data
        station1 = int(items[0])
        station2 = int(items[1])
        tubeline = int(items[2])

        #Add this edge if it doesn't already exist
        if graph.w(station1, station2) == float("Inf"):
            #Weight is the eulcidean distance between the stations
            weight = stations[station1].distance_to(stations[station2])

            graph.add_edge(station1, station2, weight)
            graph.add_edge(station2, station1, weight)

            #Initializes the list of lines that this edges has
            edge_lines[(station1, station2)] = set({})
            edge_lines[(station2, station1)] = set({})

        #Adds this line to the list of lines
        edge_lines[(station1, station2)].add(tubeline)
        edge_lines[(station2, station1)].add(tubeline)

        #Adds stations to the line
        if lines.get(tubeline) == None:
            lines[tubeline] = set({})

        lines[tubeline].add(station1)
        lines[tubeline].add(station2)

        line = file.readline()

    #Close the file stream to prevent resource leak
    file.close()

    return (graph, lines, edge_lines)

#Generates the heuristic based on the station data and desired end station
def generate_heuristic(stations: dict[int, Station], end: int, multiplier: float) -> dict[int, float]:
    heuristic = {}
    end_station = stations[end]

    #Calculate each stations distance to the end station
    for station in stations.keys():
        heuristic[station] = multiplier * stations[station].distance_to(end_station)

    return heuristic

def read_stuff():
    (empty_graph, stations) = read_stations_csv()

    print("#" * 10)
    print("Empty Graph:", empty_graph.nodes)
    print("#" * 10)
    print("Stations:", stations)

    (final_graph, lines) = read_connections_csv(empty_graph)

    print("#" * 10)
    print("Final Graph:", final_graph.nodes)
    print("#" * 10)
    print("Lines:", lines)

def calc_path_length(graph: WeightedGraph, path: list[int]) -> float:
    length = 0
    
    for src_ind in range(len(path) - 1):
        src = path[src_ind]
        dst = path[src_ind + 1]
        length += graph.w(src, dst)

    return length

#Takes in a path and returns the transfers the path takes
def calc_path_transfers(path: list[int], edge_lines: dict[tuple[int, int], set[int]]) -> int:
    if len(path) <= 1:
        return 0
    
    curLines = edge_lines[(path[0], path[1])]
    transfers = 0

    for i in range(len(path) - 1):
        newLines = edge_lines[(path[i], path[i + 1])].intersection(curLines)

        #If there is no intersection between the two lines, the path must require a transfer
        if len(newLines) == 0:
            transfers += 1
            curLines = edge_lines[(path[i], path[i + 1])]
        #Otherwise the path does not require a transfer so we limit the lines to the ones they must be on
        else:
            curLines = newLines

    return transfers


def compare_algorithms(graph, stations, edge_lines):
    station_ids = list(stations.keys())
    results = []
    time_saved = {}
    accuracy = {}
    heuristic_weights = [0.25, 0.5, 1, 2, 5]

    for weight in heuristic_weights:
        time_saved[weight] = 0
        accuracy[weight] = 0

    # Iterate over all pairs of stations
    for src in station_ids:
        for dst in station_ids:
            # Ensure src and dst are different
            if src == dst:
                continue

            # Timing for Dijkstra
            start = timeit.default_timer()
            dijkstra_result = dijkstra(graph, src, k=1000000)
            dijkstra_time = timeit.default_timer() - start
            dijkstra_transfers = calc_path_transfers(dijkstra_result[dst][1], edge_lines)

            a_star_results = []

            # Timing for A* using multiple heuristic multipliers (handling station IDs correctly)
            for weight in heuristic_weights:
                heuristic = generate_heuristic(stations, dst, weight)
            
                start = timeit.default_timer()
                _, a_star_path = A_star(graph, src, dst, heuristic)
                a_star_time = timeit.default_timer() - start
                path_length = calc_path_length(graph, a_star_path)

                a_star_results.append({
                    "heuristic_weight": weight,
                    "time": a_star_time,
                    "length": path_length,
                })

                time_saved[weight] += (dijkstra_time - a_star_time) / dijkstra_time
                accuracy[weight] += dijkstra_result[dst][0] / path_length

            # Record the results
            results.append({
                "src": src,
                "dst": dst,
                "dijkstra_time": dijkstra_time,
                "a_star_results": a_star_results,
                "dijkstra_path_length": dijkstra_result[dst][0] if dst in dijkstra_result else None,
                "shortest_path_transfers": dijkstra_transfers
            })

    for weight in heuristic_weights:
        time_saved[weight] /= len(station_ids) ** 2
        time_saved[weight] *= 100
        accuracy[weight] /= len(station_ids) ** 2
        accuracy[weight] *= 100

    return (results, time_saved, accuracy)


if __name__ == "__main__":
    # Read the station data and graph
    empty_graph, stations = read_stations_csv()
    final_graph, lines, edge_lines = read_connections_csv(empty_graph, stations)
    print("Graph and stations loaded.")

    # Run the comparison of A* and Dijkstra
    (results, time_saved, accuracy) = compare_algorithms(final_graph, stations, edge_lines)

    # Print the final results
    print("Algorithm comparison results:")
    for result in results:
        print(f"From Station {result['src']} to Station {result['dst']}:")
        print(f"  Dijkstra Time: {result['dijkstra_time']:.6f}s, Path Length: {result['dijkstra_path_length']}")
        print(f"  Transfers: {result['shortest_path_transfers']}")
        print(f"A* results:")
        for a_star_result in result['a_star_results']:
            print(f"  Heuristic-{a_star_result['heuristic_weight']} Time: {a_star_result['time']:.6f}s, Path Length: {a_star_result['length']}")
        print("-" * 50)

    print("Avg results:")
    for key in time_saved:
        print(f"A* - {key} time saved: {time_saved[key]:.6f}%")
        print(f"A* - {key} accuracy: {accuracy[key]:.6f}%")

    #Draws accuracy plot
    draw_plot(list(accuracy.values()), None, None, "Average accuracy of AStar relative to Dijkstras for multiple heuristic weightings", "Heuristic Weighting", "Avg Accuracy in %", True,  list(map(lambda x: str(x), list(accuracy.keys()))))

    #Draws time saved plot
    draw_plot(list(time_saved.values()), None, None, "Average time AStar saves relative to Dijkstras for multiple heuristic weightings", "Heuristic Weighting", "Avg Time Saved in %", True,  list(map(lambda x: str(x), list(time_saved.keys()))))

    #Collects the distribution of lines transferred
    lines_transferred = [ 0 for _ in range(100) ]

    for result in results:
        transfers = result['shortest_path_transfers']

        lines_transferred[transfers] += 1

    #Splits off trailing 0s that may be present
    for ind in range(len(lines_transferred) - 1, -1, -1):
        if lines_transferred[ind] > 0:
            lines_transferred = lines_transferred[0 : ind + 1]
            break
    
    #Graphs the distribution of line transfers
    draw_plot(lines_transferred, None, None, "Distribution for amount of path transfers required by each shortest path", "Amount of Transfers", "Amount of shortest paths", True,  [ str(i) for i in range(0, len(lines_transferred))])

    #Gets the accuracy and time_saved for each level of line transfers
    no_trans_acc = {}
    no_trans_time = {}
    adj_acc = {}
    adj_time = {}
    multi_trans_acc = {}
    multi_trans_time = {}

    #Goes through each result and adds the accuracy and time to the appropriate dictionaries
    for result in results:
        for a_star_result in result['a_star_results']:
            weight = a_star_result['heuristic_weight']
            
            accuracy = result['dijkstra_path_length'] / a_star_result['length']
            time = (result['dijkstra_time'] - a_star_result['time']) / result['dijkstra_time']

            #If no transfers, add it to the no transfers dictionaries
            if result['shortest_path_transfers'] == 0:
                if no_trans_acc.get(weight) == None:
                    no_trans_acc[weight] = accuracy
                    no_trans_time[weight] = time
                else:
                    no_trans_acc[weight] += accuracy
                    no_trans_time[weight] += time

            #If one transfer, add it to the adjacent line dictionaries
            elif result['shortest_path_transfers'] == 1:
                if adj_acc.get(weight) == None:
                    adj_acc[weight] = accuracy
                    adj_time[weight] = time
                else:
                    adj_acc[weight] += accuracy
                    adj_time[weight] += time

            #If multiple transfers, add it to the multi transfers dictionaries
            else:
                if multi_trans_acc.get(weight) == None:
                    multi_trans_acc[weight] = accuracy
                    multi_trans_time[weight] = time
                else:
                    multi_trans_acc[weight] += accuracy
                    multi_trans_time[weight] += time

    #Converts the sum for each weight to the average, based on the amount in each section from the lines transferred array
    for weight in no_trans_acc.keys():
        no_trans_acc[weight] /= (lines_transferred[0] / 100)
        no_trans_time[weight] /= (lines_transferred[0] / 100)
        adj_acc[weight] /= (lines_transferred[1] / 100)
        adj_time[weight] /= (lines_transferred[1] / 100)
        multi_trans_acc[weight] /= (sum(lines_transferred[2:]) / 100)
        multi_trans_time[weight] /= (sum(lines_transferred[2:]) / 100)

    #Draws accuracy plot for no transfers
    draw_plot(list(no_trans_acc.values()), None, None, "Average accuracy of AStar relative to Dijkstras for multiple heuristic weightings with no transfers", "Heuristic Weighting", "Avg Accuracy in %", True,  list(map(lambda x: str(x), list(no_trans_acc.keys()))))

    #Draws time saved plot for no transfers
    draw_plot(list(no_trans_time.values()), None, None, "Average time AStar saves relative to Dijkstras for multiple heuristic weightings with no transfers", "Heuristic Weighting", "Avg Time Saved in %", True,  list(map(lambda x: str(x), list(no_trans_time.keys()))))

    #Draws accuracy plot for adjacent lines
    draw_plot(list(adj_acc.values()), None, None, "Average accuracy of AStar relative to Dijkstras for multiple heuristic weightings with adjacent lines", "Heuristic Weighting", "Avg Accuracy in %", True,  list(map(lambda x: str(x), list(adj_acc.keys()))))

    #Draws time saved plot for adjacent lines
    draw_plot(list(adj_time.values()), None, None, "Average time AStar saves relative to Dijkstras for multiple heuristic weightings with adjacent lines", "Heuristic Weighting", "Avg Time Saved in %", True,  list(map(lambda x: str(x), list(adj_time.keys()))))

    #Draws accuracy plot for multiple transfers
    draw_plot(list(multi_trans_acc.values()), None, None, "Average accuracy of AStar relative to Dijkstras for multiple heuristic weightings with multiple transfers", "Heuristic Weighting", "Avg Accuracy in %", True,  list(map(lambda x: str(x), list(multi_trans_acc.keys()))))

    #Draws time saved plot for multiple transfers
    draw_plot(list(multi_trans_time.values()), None, None, "Average time AStar saves relative to Dijkstras for multiple heuristic weightings with multiple transfers", "Heuristic Weighting", "Avg Time Saved in %", True,  list(map(lambda x: str(x), list(multi_trans_time.keys()))))