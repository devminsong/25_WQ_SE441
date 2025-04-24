
"""
# ===============================================
#  Created by: Kenny Davila Castellanos
#              for CSC 380/480
#
#  MODIFIED BY: [MIN SONG]
# ===============================================
"""
from .state import State
from .map import CityMap
from .search_request import SearchRequest
from typing import List, Tuple

"""
    The Problem class. This is where most of the problem-specific logic (e.g. 
    the problem formulation) is done. For this problem in particular, we need
    two attributes: the city map and the test case (a Search Request). The 
    city map contains all the information related to the underlying navigation
    problem, while the Search Request includes all the info related to one
    specific search (e.g. starting point, targets locations, etc.). It is the 
    job of the Problem class to put these two together to generate the states,
    actions, transitions, costs and heuristics needed for search.       
"""


class Problem:
    """
        The constructor. A new instance of the Problem class should be used
        for every new search that will be executed.
    """
    def __init__(self, map: CityMap, test_case: SearchRequest):
        self.__city_map = map
        self.__current_case = test_case

    def get_city_map(self):
        return self.__city_map

    def get_current_case(self):
        return self.__current_case

    """
        Use the information from map and test case to generate the 
        Initial State for the current search problem.
        
        Hint: current_case has information about both the starting location
              and the search nodes. 
    """
    def get_initial_state(self) -> State | None:
        start_location = self.get_current_case().get_start_location()
        target_locations = self.get_current_case().get_targets()

        visited_targets = []
        for _ in range(len(target_locations)):
            visited_targets.append(False)
            
        return State(start_location, visited_targets)

    """
        Check if the given state object represent a goal state. This can be
        determined using information stored in the state itself and also
        in the current_case object. 
        
        Hint: Keep in mind that the goal state requires the driver to be back 
            at the same location where the trip start, and all targets should 
            have been visited.
    """
    def is_goal_state(self, state: State) -> bool:
        current_location = state.get_location()
        start_location = self.get_current_case().get_start_location()
        visited_targets = state.get_visited_targets()
        
        is_back = (current_location == start_location)
        
        all_targets_visited = True
        for visited in visited_targets:
            if not visited:
                all_targets_visited = False
                break

        return is_back and all_targets_visited

    """
        Given a state, this function generates a List of the Children states.
        This function basically implements the transition model. First, you 
        need to consider which locations are neighbors of current location 
        (the map class can help with that). Actions here are equal to travel
        to any of these immediate neighbors. For each of these actions, you 
        will have to generate a new state object. Keep in mind that some of 
        these locations might be part of the targets. As such, your state 
        generation process needs to correctly update the vector of visited
        nodes for the child state. 
        
        Hint: This function needs to use the city_map, the state, and the
            search_case. 
        Hint: Mind the alignment between target locations in the Search Request
            object and visited targets in the State object.  
    """
    def generate_children(self, state: State) -> List[State]:
        current_location = state.get_location()
        neighbor_locations = self.get_city_map().get_neighbors(current_location)
      
        visited_targets = state.get_visited_targets()
        target_locations = self.get_current_case().get_targets()
        child_states = []
        for neighbor_location in neighbor_locations:
            child_visited_targets = list(visited_targets)

            for i, target in enumerate(target_locations):
                if neighbor_location == target and not visited_targets[i]:
                    child_visited_targets[i] = True
                    break

            child_state = State(neighbor_location, child_visited_targets)
            child_states.append(child_state)

        return child_states

    """
        Cost-Model: cost of executing the given action on the given state
              cost = Action-Cost( State, Action)
    
        Hint 1: You need to consider the location of current state
        Hint 2: The city map can give you the info you need about costs of
             traveling between neighboring locations.
        Hint 3: Do not confuse this with the straight line traveling distance 
             used to estimate the cost of traveling between non-neighboring 
             locations    
    """
    def get_action_cost(self, state: State, action: str) -> float:
        current_location = state.get_location()
        next_location = action
        cost = self.get_city_map().get_cost(current_location, next_location)
        return cost

    """
        Cost-Estimation-Model: estimated cost of reaching a goal state from the
        given state. This is the Heuristic function required by A*
    
            estimated-cost = Heuristic( State)
    
        Hint 1: You need to consider the location of current state
        Hint 2: The city map can help you estimate the lower-bound of the real
            cost of traveling certain distances.
        Hint 3: The write-up offers a more detailed explanation of the function
            that you should be implementing here
    
    """
    def estimate_cost_to_solution(self, state: State) -> float:
        l_current = state.get_location()
        l_final = self.get_current_case().get_start_location()

        targets = self.get_current_case().get_targets()
        pending_targets = []
        for i, visited in enumerate(state.get_visited_targets()):
            if not visited:
                pending_targets.append(targets[i])

        if not pending_targets:
            return self.get_city_map().get_straight_line_distance(l_current, l_final)
        else:
            l_close = None
            l_far = None
            min_distance = float('inf')
            max_distance = float('-inf')

            for pending_target in pending_targets:
                distance_to_final = self.get_city_map().get_straight_line_distance(pending_target, l_final)
                if distance_to_final < min_distance:
                    min_distance = distance_to_final
                    l_close = pending_target
                if distance_to_final > max_distance:
                    max_distance = distance_to_final
                    l_far = pending_target

            c_beginning = min(
                self.get_city_map().get_straight_line_distance(l_current, l_close),
                self.get_city_map().get_straight_line_distance(l_current, l_far)
            )

            c_middle = self.get_city_map().get_straight_line_distance(l_close, l_far)
            c_final = self.get_city_map().get_straight_line_distance(l_close, l_final)

            return c_beginning + c_middle + c_final