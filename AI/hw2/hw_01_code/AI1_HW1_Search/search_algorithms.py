
"""
# ===============================================
#  Created by: Kenny Davila Castellanos
#              for CSC 380/480
#
#  MODIFIED BY: [MIN SONG]
# ===============================================
"""

import time
import heapq

from .search_result import SearchResults
from .search_tree_node import SearchTreeNode

from AI1_HW_Problem.problem import Problem
from AI1_HW_Problem.search_request import SearchRequest

class SearchAlgorithms:
    BreadthFirstSearch = 0
    UniformCostSearch = 1
    AStarSearch = 2

    """
        Implementation of the Breadth-first Search (BFS) algorithm. The only input
        required is the problem, which includes a reference to current CityMap and 
        current Search Request. The function needs to compute and store search 
        results and other statistics inside of a SearchResults object. 
    """ 
    @staticmethod
    def breadth_first_search(problem: Problem) -> SearchResults:
        start_node = SearchTreeNode(None, None, problem.get_initial_state(), 0)

        if problem.is_goal_state(start_node.get_state()):
            return SearchResults(start_node.path_to_root(), start_node.get_path_cost(), 1, 1)

        frontier = [start_node]
        reached = {start_node.get_state().get_representation()}

        nodes_explored = 0
        nodes_reached = 1

        while frontier:
            current_node = frontier.pop(0)
            nodes_explored += 1            
            
            child_states = problem.generate_children(current_node.get_state())
            for child_state in child_states:
                action = child_state.get_location()
                action_cost = problem.get_action_cost(current_node.get_state(), action)
                child_node = SearchTreeNode(current_node, action, child_state, current_node.get_path_cost() + action_cost)                    
                s = child_state.get_representation()

                if problem.is_goal_state(child_node.get_state()):
                    return SearchResults(child_node.path_to_root(), child_node.get_path_cost(), nodes_reached, nodes_explored)

                if s not in reached:
                    reached.add(s)
                    nodes_reached += 1  
                    frontier.append(child_node)

        return SearchResults(None, None, nodes_reached, nodes_explored)

    """
        Implementation of the Uniform Cost Search (UCS) algorithm. The only input
        required is the problem, which includes a reference to current CityMap and 
        current Search Request. The function needs to compute and store search 
        results and other statistics inside of a SearchResults object. 
    """
    @staticmethod
    def uniform_cost_search(problem: Problem) -> SearchResults:        
        start_node = SearchTreeNode(None, None, problem.get_initial_state(), 0)

        if problem.is_goal_state(start_node.get_state()):
            return SearchResults(start_node.path_to_root(), start_node.get_path_cost(), 1, 1)

        frontier = [(0, start_node)]
        reached = {start_node.get_state().get_representation(): 0}

        nodes_explored = 0
        nodes_reached = 1

        while frontier:
            path_cost, current_node = heapq.heappop(frontier)
            nodes_explored += 1

            if problem.is_goal_state(current_node.get_state()):
                return SearchResults(current_node.path_to_root(), path_cost, nodes_reached, nodes_explored)

            child_states = problem.generate_children(current_node.get_state())
            for next_state in child_states:
                action = next_state.get_location()
                action_cost = problem.get_action_cost(current_node.get_state(), action)
                child_path_cost = path_cost + action_cost
                s = next_state.get_representation()

                if s not in reached or child_path_cost < reached[s]:
                    reached[s] = child_path_cost
                    nodes_reached += 1
                    child_node = SearchTreeNode(current_node, action, next_state, child_path_cost)
                    heapq.heappush(frontier, (child_path_cost, child_node))

        return SearchResults(None, None, nodes_reached, nodes_explored)

    """
        Implementation of the A* Search algorithm. The only input
        required is the problem, which includes a reference to current CityMap and 
        current Search Request. The function needs to compute and store search 
        results and other statistics inside of a SearchResults object. 
    """
    @staticmethod
    def A_start_search(problem: Problem) -> SearchResults:
        initial_state = problem.get_initial_state()
        if initial_state is None:
            return SearchResults(None, None, 0, 0)

        start_node = SearchTreeNode(None, None, problem.get_initial_state(), 0)

        if problem.is_goal_state(start_node.get_state()):
            return SearchResults(start_node.path_to_root(), start_node.get_path_cost(), 1, 1)

        h = problem.estimate_cost_to_solution(initial_state)
        frontier = [(h, 0, start_node)]
        reached = {initial_state.get_representation(): h}

        nodes_explored = 0
        nodes_reached = 1

        while frontier:
            f, path_cost, current_node = heapq.heappop(frontier)
            nodes_explored += 1

            if problem.is_goal_state(current_node.get_state()):
                return SearchResults(current_node.path_to_root(), path_cost, nodes_reached, nodes_explored)

            child_states = problem.generate_children(current_node.get_state())
            for child_state in child_states:
                action = child_state.get_location()
                action_cost = problem.get_action_cost(current_node.get_state(), action)
                child_path_cost = path_cost + action_cost
                s = child_state.get_representation()

                child_h = problem.estimate_cost_to_solution(child_state)
                child_f = child_path_cost + child_h
                child_node = SearchTreeNode(current_node, action, child_state, child_path_cost)

                if s not in reached or child_f < reached[s]:
                    nodes_reached += 1
                    reached[s] = child_f
                    heapq.heappush(frontier, (child_f, child_path_cost, child_node))
     
        return SearchResults(None, None, nodes_reached, nodes_explored)

    """
        Auxiliary function for printing search results 
    """
    @staticmethod
    def print_solution_details(test_case: SearchRequest, search_results: SearchResults, search_time: float):
        if search_results.get_solution() is None:
            print(" --> No Solution was found!")
        else:
            # this part is just for formatting... with highlight for important nodes in the path
            tempo_list = []
            for value in [test_case.get_start_location()] + search_results.get_solution():
                if value == test_case.get_start_location():
                    tempo_list.append(f"[{value}]")
                elif value in test_case.get_targets():
                    tempo_list.append(f"({value})")
                else:
                    tempo_list.append(value)
            full_path = " -> ".join(tempo_list)
            print(f" --> Solution found: {full_path}")
            print(f" --> Solution Cost: {search_results.get_cost()}")
        print(f" --> Nodes Reached: {search_results.reached_nodes_count()}")
        print(f" --> Nodes Explored: {search_results.explored_nodes_count()}")
        print(f" --> Search Time (s): {search_time}")

        return full_path

    """
        Auxiliary Function for running the search algorithm specified, 
        and printing the results and statistics.
    """
    @staticmethod
    def search(problem: Problem, algorithm: int) -> tuple[SearchResults | None, float]:
        # Note: This code might look awkward, but this is intentional
        # the idea is to NOT count the print operation as part of the search time
        if algorithm == SearchAlgorithms.BreadthFirstSearch:
            print("\n - Running Breadth-First Search")
        elif algorithm == SearchAlgorithms.UniformCostSearch:
            print("\n - Running Uniform Cost Search")
        elif algorithm == SearchAlgorithms.AStarSearch:
            print("\n - Running A Star Search")
        else:
            raise Exception(f"Invalid Search Algorithm: {algorithm}")

        start_time = time.time()
        if algorithm == SearchAlgorithms.BreadthFirstSearch:
            solution = SearchAlgorithms.breadth_first_search(problem)
        elif algorithm == SearchAlgorithms.UniformCostSearch:
            solution = SearchAlgorithms.uniform_cost_search(problem)
        else:
            # by default, assume it is the A* algorithm
            solution = SearchAlgorithms.A_start_search(problem)
        end_time = time.time()
        total_time = end_time - start_time

        # SearchAlgorithms.print_solution_details(problem.get_current_case(), solution, total_time)

        return solution, total_time