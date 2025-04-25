
"""
# ===============================================
#  Created by: Kenny Davila Castellanos
#              for CSC 380/480
#
#  MODIFIED BY: [MIN SONG]
# ===============================================
"""

import sys
import os
import statistics
import csv

from AI1_HW_Problem.map import CityMap
from AI1_HW_Problem.search_request import SearchRequest
from AI1_HW_Problem.problem import Problem
from AI1_HW1_Search.search_algorithms import SearchAlgorithms

"""
    This function provides a few examples of how to use the CityMap class
"""
def example_using_map_class(map: CityMap):
    # TODO: feel free to modify and/or remove this code
    # ... Checking the name of the map
    print(f"City Name: {map.get_name()}")
    # ... Checking the locations
    print(f"Locations: {map.get_locations()}")
    # ... This will give you the neighbors of any location in the map
    loc = 'F3'
    print(f"Neighbors of {loc}: {map.get_neighbors(loc)}")
    # ... This will give you the cost of an existing connection
    other = 'F4'
    print(f"Cost of traveling from {loc} to {other}: {map.get_cost(loc, other)}")
    # ... and will return null if there is no connection between the nodes ...
    other = 'A1'
    print(f"Cost of traveling from {loc} to {other}: {map.get_cost(loc, other)}")
    # ... but you can still get an estimation of the cost based on the straight line distance
    #     even when there is no direct connection between these points
    print(f"Straight Line distance from {loc} to {other}: {map.get_straight_line_distance(loc, other)}")



def main():
    # ... loading the map from JSON file ...
    map = CityMap.FromFile("./tegucigalpa.json")
    # ... loading the test cases from JSON file ...
    test_cases = SearchRequest.FromTestCasesFile("./test_cases.json")
    # TODO: Check the code for this function to understand how the map class works
    #       Afterwards, modify  or remove this line of code
    # example_using_map_class(map)

    # TODO: Check the code for this function to understand how the test case class works
    #       Afterwards, modify or remove this line of code

    for test_case in test_cases:
        print("\n\nTest Case info:")
        print(f" - Name: {test_case.get_name()}")
        print(f" - Starting Location: {test_case.get_start_location()}")
        print(f" - Delivery Locations: {test_case.get_targets()}")

        # Create the problem
        problem = Problem(map, test_case)

        # # use BFS ....
        # SearchAlgorithms.search(problem, SearchAlgorithms.BreadthFirstSearch)

        # # use UCS ....
        # SearchAlgorithms.search(problem, SearchAlgorithms.UniformCostSearch)

        # # use A* ....
        # SearchAlgorithms.search(problem, SearchAlgorithms.AStarSearch)
        
        multiple_search(problem, SearchAlgorithms.BreadthFirstSearch, "BFS", 1)
        multiple_search(problem, SearchAlgorithms.BreadthFirstSearch, "UCS", 1)
        multiple_search(problem, SearchAlgorithms.BreadthFirstSearch, "A*", 1)

    # write_to_csv(map, test_cases)

def multiple_search(problem: Problem, algorithm: int, algorithm_name: str, num_runs: int = 5) -> tuple[list, list, list, list]:
    solution_costs = []
    reached_node_counts = []
    explored_node_counts = []
    search_times = []

    print(f"\n - Running {algorithm_name} {num_runs} times")

    for i in range(num_runs):
        print(f"\n run {i}")
        search_result, search_time = SearchAlgorithms.search(problem, algorithm)

        if search_result:
            solution_costs.append(search_result.get_cost() if search_result.get_solution() else None)
            reached_node_counts.append(search_result.reached_nodes_count())
            explored_node_counts.append(search_result.explored_nodes_count())
        else:
            solution_costs.append(None)
            reached_node_counts.append(0)
            explored_node_counts.append(0)

        search_times.append(search_time)
        full_path = SearchAlgorithms.print_solution_details(problem.get_current_case(), search_result, search_time)

    return solution_costs, reached_node_counts, explored_node_counts, search_times, full_path

def write_to_csv(map: CityMap, test_cases: list[SearchRequest]):
    with open('search_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Test Case', 'Algorithm', 'Solution Cost (mi)', 'Nodes Reached', 'Nodes Explored', 'Search Time (s)', 'Solution Found'])

        for test_case in test_cases:
            test_case_name = test_case.get_name()
            print(f"\n\n--- Running searches for Test Case: {test_case_name} ---")
            print(f" - Name: {test_case.get_name()}")
            print(f" - Starting Location: {test_case.get_start_location()}")
            print(f" - Delivery Locations: {test_case.get_targets()}")

            problem = Problem(map, test_case)
            num_runs = 1

            solution_costs, reached_node_counts, explored_node_counts, search_times, full_path = multiple_search(
                problem, SearchAlgorithms.BreadthFirstSearch, "BFS", num_runs
            )
            median_cost = statistics.median([cost for cost in solution_costs if cost is not None]) if solution_costs else "N/A"
            median_reached = statistics.median(reached_node_counts) if reached_node_counts else 0
            median_explored = statistics.median(explored_node_counts) if explored_node_counts else 0
            median_time = statistics.median(search_times) if search_times else 0
            writer.writerow([test_case_name, "BFS", median_cost, median_reached, median_explored, f"{median_time:.6f}", full_path])

            solution_costs, reached_node_counts, explored_node_counts, search_times, full_path = multiple_search(
                problem, SearchAlgorithms.UniformCostSearch, "UCS", num_runs
            )
            median_cost = statistics.median([cost for cost in solution_costs if cost is not None]) if solution_costs else "N/A"
            median_reached = statistics.median(reached_node_counts) if reached_node_counts else 0
            median_explored = statistics.median(explored_node_counts) if explored_node_counts else 0
            median_time = statistics.median(search_times) if search_times else 0
            writer.writerow([test_case_name, "UCS", median_cost, median_reached, median_explored, f"{median_time:.6f}", full_path])

            solution_costs, reached_node_counts, explored_node_counts, search_times, full_path = multiple_search(
                problem, SearchAlgorithms.AStarSearch, "A*", num_runs
            )
            median_cost = statistics.median([cost for cost in solution_costs if cost is not None]) if solution_costs else "N/A"
            median_reached = statistics.median(reached_node_counts) if reached_node_counts else 0
            median_explored = statistics.median(explored_node_counts) if explored_node_counts else 0
            median_time = statistics.median(search_times) if search_times else 0
            writer.writerow([test_case_name, "A*", median_cost, median_reached, median_explored, f"{median_time:.6f}", full_path])

if __name__ == "__main__":
    main()
