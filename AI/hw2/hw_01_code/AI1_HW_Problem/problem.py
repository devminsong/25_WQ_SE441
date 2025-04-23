
"""
# ===============================================
#  Created by: Kenny Davila Castellanos
#              for CSC 380/480
#
#  MODIFIED BY: [Your NAME]
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
        start_location = self.__current_case.get_start_location()
        target_locations = self.__current_case.get_targets()
        # 초기 상태에서는 아직 아무 목표도 방문하지 않았으므로 False로 초기화된 리스트를 생성합니다.
        initial_visited_targets = [False] * len(target_locations)
        # State 객체를 생성하여 반환합니다.
        initial_state = State(start_location, initial_visited_targets)
        return initial_state

    """
        Check if the given state object represent a goal state. This can be
        determined using information stored in the state itself and also
        in the current_case object. 
        
        Hint: Keep in mind that the goal state requires the driver to be back 
            at the same location where the trip start, and all targets should 
            have been visited.
    """
    def is_goal_state(self, state: State) -> bool:
        # 현재 상태의 위치를 가져옵니다. [1, 2]
        current_location = state.get_location()
        # 검색 요청의 시작 위치를 가져옵니다. [3, 4]
        start_location = self.__current_case.get_start_location()
        # 현재 상태에서 방문한 목표 상태 리스트를 가져옵니다. [1, 2]
        visited_targets = state.get_visited_targets()
        # 검색 요청의 목표 위치 리스트를 가져옵니다. [3, 5]
        target_locations = self.__current_case.get_targets()

        # 1. 현재 위치가 시작 위치와 동일한지 확인합니다. [6]
        is_at_start = (current_location == start_location)

        # 2. 모든 목표 위치를 방문했는지 확인합니다. [1, 6, 7]
        all_targets_visited = all(visited_targets)

        # 위 두 조건이 모두 참이면 목표 상태입니다.
        print(f'is_goal_state 확인 : {current_location} {start_location} {visited_targets}')
        return is_at_start and all_targets_visited

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
        children = []
        current_location = state.get_location()
        visited_targets = state.get_visited_targets()
        target_locations = self.__current_case.get_targets()

        # 현재 위치의 이웃 노드들을 가져옵니다.
        neighbor_locations = self.__city_map.get_neighbors(current_location)

        # 각 이웃 노드에 대해 새로운 자식 상태를 생성합니다.
        for neighbor_location in neighbor_locations:
            # 새로운 자식 상태의 방문 상태 리스트를 복사합니다.
            new_visited_targets = list(visited_targets)

            # 이웃 위치가 목표 위치 중 하나인지 확인하고, 아직 방문하지 않았다면 방문 상태를 업데이트합니다.
            for i, target in enumerate(target_locations):
                if neighbor_location == target and not visited_targets[i]:
                    new_visited_targets[i] = True
                    break  # 하나의 목표만 방문했으므로 루프를 종료합니다.

            # 새로운 상태 객체를 생성합니다.
            print(f'{current_location}의 이웃 노드 {neighbor_location} 생성')
            new_state = State(neighbor_location, new_visited_targets)
            children.append(new_state)

        print(f"""자식state 생성:
              현재 state는 {current_location}이고
              visited 상태는 {visited_targets}이고
              target은 {target_locations}이고
              현재 neighbor는 {neighbor_locations}이고
              갱신된 visited는 {new_visited_targets}입니다.
        """)
        return children

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
        # 현재 상태의 위치를 가져옵니다 [3].
        current_location = state.get_location()
        # action은 다음 위치를 나타냅니다.
        next_location = action
        # CityMap 객체를 사용하여 현재 위치에서 다음 위치로 이동하는 비용(거리)을 가져옵니다 [3].
        cost = self.__city_map.get_cost(current_location, next_location)
        # 연결이 없는 경우 (None 반환), 이는 발생해서는 안 되므로 예외를 발생시키거나 적절한 처리를 할 수 있습니다.
        # generate_children에서 이웃한 위치에 대해서만 action이 생성되므로 None이 반환될 가능성은 낮습니다.
        if cost is None:
            raise ValueError(f"No direct connection found between {current_location} and {next_location}")
        # 계산된 비용을 반환합니다 [3].
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
        # 현재 상태의 위치를 가져옵니다.
        current_location = state.get_location()
        # 탐색 요청의 시작 위치 (이자 최종 목표 위치)를 가져옵니다.
        final_location = self.__current_case.get_start_location()
        # 탐색 요청의 목표 지점 목록을 가져옵니다.
        targets = self.__current_case.get_targets()
        # 현재 상태에서 방문하지 않은 목표 지점들을 식별합니다.
        pending_targets = [targets[i] for i, visited in enumerate(state.get_visited_targets()) if not visited]

        # 방문해야 할 목표 지점이 없는 경우, 현재 위치에서 시작 위치로 돌아가는 직선 거리를 추정합니다.
        if not pending_targets:
            return self.__city_map.get_straight_line_distance(current_location, final_location)
        else:
            # 방문해야 할 목표 지점이 하나 이상인 경우, 휴리스틱 비용을 계산합니다.
            # 최종 위치(시작 위치)에서 가장 가깝고 먼 미방문 목표 지점을 찾습니다.
            closest_target = None
            farthest_target = None
            min_distance_to_final = float('inf')
            max_distance_to_final = float('-inf')

            for target in pending_targets:
                distance_to_final = self.__city_map.get_straight_line_distance(target, final_location)
                if distance_to_final < min_distance_to_final:
                    min_distance_to_final = distance_to_final
                    closest_target = target
                if distance_to_final > max_distance_to_final:
                    max_distance_to_final = distance_to_final
                    farthest_target = target

            # 시작 부분 비용 추정: 현재 위치에서 가장 가깝거나 먼 미방문 목표 지점 중 더 가까운 곳으로 이동하는 비용
            cost_beginning = min(
                self.__city_map.get_straight_line_distance(current_location, closest_target),
                self.__city_map.get_straight_line_distance(current_location, farthest_target)
            )

            # 중간 부분 비용 추정: 가장 가까운 미방문 목표 지점에서 가장 먼 미방문 목표 지점으로 이동하는 비용
            cost_middle = self.__city_map.get_straight_line_distance(closest_target, farthest_target)

            # 마지막 부분 비용 추정: 가장 가까운 미방문 목표 지점에서 최종 위치로 이동하는 비용
            cost_final = self.__city_map.get_straight_line_distance(closest_target, final_location)

            # 총 예상 비용을 반환합니다.
            return cost_beginning + cost_middle + cost_final
