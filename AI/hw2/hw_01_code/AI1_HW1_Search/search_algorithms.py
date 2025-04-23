
"""
# ===============================================
#  Created by: Kenny Davila Castellanos
#              for CSC 380/480
#
#  MODIFIED BY: [Your NAME]
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
        # 시작 노드를 생성합니다.
        initial_state = problem.get_initial_state()
        if initial_state is None:
            return SearchResults(None, None, 0, 0)
        start_node = SearchTreeNode(None, None, initial_state, 0)

        # 목표 상태에 도달했는지 확인합니다.
        if problem.is_goal_state(start_node.get_state()):
            return SearchResults(start_node.path_to_root(), start_node.get_path_cost(), 1, 1)

        # FIFO 큐를 사용하여 frontier를 초기화하고 시작 노드를 추가합니다 [4, 5].
        frontier = [start_node]
        # 이미 도달한 상태를 추적하기 위한 집합을 초기화하고 시작 상태를 추가합니다 [5].
        reached = {initial_state.get_representation()}
        print(f'frontier: {frontier} reached: {reached}')

        # 탐색된 노드와 도달한 노드의 수를 초기화합니다.
        nodes_explored = 0
        nodes_reached = 1

        # frontier가 비어 있지 않은 동안 탐색을 계속합니다 [5].
        while frontier:
            # frontier에서 가장 오래된 노드를 꺼냅니다 (FIFO) [4, 5].
            current_node = frontier.pop(0)
            print("")
            print("################ POP #################")
            print(f'frontier에서 pop한 노드: {current_node}')
            print("################ POP #################")
            print("")

            nodes_explored += 1

            # 현재 상태를 기반으로 가능한 자식 상태들을 생성합니다.
            # 이제 problem.generate_children 함수는 다음 가능한 모든 상태를 반환합니다.
            next_states = problem.generate_children(current_node.get_state())
            print(f'while frontier에서 next state count : {len(next_states)}')
            if next_states:
                for next_state in next_states:
                    print('')
                    s = next_state.get_representation()
                    # print(f'generate_children이 리턴한 다음 state는: {s}')

                    # 자식 상태가 아직 도달하지 않은 상태라면 [5]:
                    if s not in reached:
                        nodes_reached += 1
                        reached.add(s)
                        # 새로운 노드를 생성하고 frontier에 추가합니다 [5].
                        # action은 현재 노드의 상태에서 다음 상태로 이동하기 위해 수행된 액션입니다.
                        # 이는 generate_children 함수를 통해 생성된 next_state가 어떤 이웃 위치로 이동한 결과인지 알면 됩니다.
                        # 따라서, current_node의 위치와 next_state의 위치를 비교하여 action을 파악할 수 있습니다.
                        action = next_state.get_location() # 이동한 위치 자체가 action이 됩니다 [1, 2].
                        print(f'action은: {action}')
                        cost = problem.get_action_cost(current_node.get_state(), action)
                        child_node = SearchTreeNode(current_node, action, next_state,
                                                    current_node.get_path_cost() + cost)
                        # 자식 노드가 목표 상태인지 확인합니다 [5].
                        if problem.is_goal_state(child_node.get_state()):
                            return SearchResults(child_node.path_to_root(), child_node.get_path_cost(),
                                                   nodes_reached, nodes_explored)
                        frontier.append(child_node)
                        print(f'{s} frontier.append 후 frontier 상태 {frontier}')
                        print(f'{s} frontier.append 후 reached 상태 {reached}')
                        print('')
                    else:
                        print(f'{s}는 reached에 있으므로 추가하지 않습니다. frontirer 상태: {frontier}')
                        print(f'{s}는 reached에 있으므로 추가하지 않습니다. reached 상태: {reached}')

        # 목표 상태에 도달하지 못하면 실패를 반환합니다 [5].
        print('탐색실패')
        return SearchResults(None, None, nodes_reached, nodes_explored)

    """
        Implementation of the Uniform Cost Search (UCS) algorithm. The only input
        required is the problem, which includes a reference to current CityMap and 
        current Search Request. The function needs to compute and store search 
        results and other statistics inside of a SearchResults object. 
    """
    @staticmethod
    def uniform_cost_search(problem: Problem) -> SearchResults:
        # 시작 상태를 얻습니다 [7].
        initial_state = problem.get_initial_state()
        if initial_state is None:
            return SearchResults(None, None, 0, 0)

        # 시작 노드를 생성합니다. 경로 비용은 0입니다 [8].
        start_node = SearchTreeNode(None, None, initial_state, 0)

        # 목표 상태인지 확인합니다 [9].
        if problem.is_goal_state(start_node.get_state()):
            return SearchResults(start_node.path_to_root(), start_node.get_path_cost(), 1, 1)

        # 우선순위 큐를 사용하여 frontier를 초기화하고 시작 노드를 추가합니다.
        # 우선순위는 경로 비용입니다 [1, 4].
        frontier = [(0, start_node)]  # (경로 비용, 노드) 튜플로 저장하여 heapq에서 우선순위를 사용하도록 합니다.

        # 이미 도달한 상태와 해당 상태의 최소 비용을 저장하는 딕셔너리를 초기화합니다.
        # (상태 표현: 최소 경로 비용) 형태입니다 [4, 10].
        reached = {initial_state.get_representation(): 0}
        print(f'frontier: {frontier} reached: {reached}')

        # 탐색된 노드와 도달한 노드의 수를 초기화합니다.
        nodes_explored = 0
        nodes_reached = 1

        # frontier가 비어 있지 않은 동안 탐색을 계속합니다 [4].
        while frontier:
            # 우선순위 큐에서 가장 낮은 경로 비용을 가진 노드를 꺼냅니다 [4, 11].
            path_cost, current_node = heapq.heappop(frontier)
            print("")
            print("################ POP #################")
            print(f'frontier에서 pop한 노드: {path_cost}, {current_node}')
            print("################ POP #################")
            print("")

            nodes_explored += 1

            # 현재 노드의 상태가 목표 상태인지 확인합니다 [4, 9].
            if problem.is_goal_state(current_node.get_state()):
                print(f'목표 상태를 찾았습니다 : {current_node.get_state()}')
                return SearchResults(current_node.path_to_root(), path_cost, nodes_reached, nodes_explored)

            # 현재 상태를 기반으로 가능한 자식 상태들을 생성합니다 [6, 12].
            next_states = problem.generate_children(current_node.get_state())
            print(f'while frontier에서 next state count : {len(next_states)}')

            if next_states:
                for next_state in next_states:
                    print('')
                    # 자식 상태로 이동하는 액션을 얻습니다. 이는 자식 상태의 위치입니다 [13].
                    action = next_state.get_location()
                    print(f'action은: {action}')

                    # 부모 노드에서 자식 노드로 이동하는 비용을 계산합니다 [6, 14].
                    cost = problem.get_action_cost(current_node.get_state(), action)
                    # 새로운 경로 비용을 계산합니다 [6].
                    new_path_cost = path_cost + cost
                    # 자식 상태의 표현을 얻습니다 [15].
                    s = next_state.get_representation()

                    # 자식 상태가 아직 도달하지 않았거나, 더 낮은 비용으로 도달할 수 있는 경우 [5, 6]:
                    if s not in reached or new_path_cost < reached[s]:
                        nodes_reached += 1
                        reached[s] = new_path_cost
                        # 새로운 자식 노드를 생성합니다 [6, 8].
                        child_node = SearchTreeNode(current_node, action, next_state, new_path_cost)
                        # 우선순위 큐에 (새로운 경로 비용, 자식 노드) 형태로 추가합니다 [6, 11].
                        heapq.heappush(frontier, (new_path_cost, child_node))
                        print(f'{s} frontier.heappush 후 frontier 상태 {frontier}')
                        print(f'{s} frontier.heappush 후 reached 상태 {reached}')
                    else:
                        print(f'{s}는 reached에 있으므로 추가하지 않습니다. frontirer 상태: {frontier}')
                        print(f'{s}는 reached에 있으므로 추가하지 않습니다. reached 상태: {reached}')

        # 목표 상태에 도달하지 못하면 실패를 반환합니다 [16].
        return SearchResults(None, None, nodes_reached, nodes_explored)

    """
        Implementation of the A* Search algorithm. The only input
        required is the problem, which includes a reference to current CityMap and 
        current Search Request. The function needs to compute and store search 
        results and other statistics inside of a SearchResults object. 
    """
    @staticmethod
    def A_start_search(problem: Problem) -> SearchResults:
        # 시작 상태를 얻습니다 [3].
        initial_state = problem.get_initial_state()
        if initial_state is None:
            return SearchResults(None, None, 0, 0)

        # 시작 노드를 생성합니다. 경로 비용은 0입니다 [3].
        start_node = SearchTreeNode(None, None, initial_state, 0)

        # 목표 상태인지 확인합니다 [4].
        if problem.is_goal_state(start_node.get_state()):
            return SearchResults(start_node.path_to_root(), start_node.get_path_cost(), 1, 1)

        # 우선순위 큐를 사용하여 frontier를 초기화하고 시작 노드를 추가합니다.
        # 우선순위는 f(n) = g(n) + h(n) 입니다 [2].
        start_heuristic = problem.estimate_cost_to_solution(initial_state)
        frontier = [(start_heuristic, 0, start_node)] # (f(n), g(n), 노드) 튜플로 저장하여 heapq에서 우선순위를 사용합니다. g(n)을 포함하여 휴리스틱 값이 같을 경우 경로 비용이 낮은 노드를 먼저 탐색하도록 합니다.

        # 이미 도달한 상태와 해당 상태의 최소 f(n) 값을 저장하는 딕셔너리를 초기화합니다.
        # (상태 표현: 최소 f(n) 값) 형태입니다 [4].
        reached = {initial_state.get_representation(): start_heuristic}
        print(f'frontier: {frontier} reached: {reached}')

        # 탐색된 노드와 도달한 노드의 수를 초기화합니다.
        nodes_explored = 0
        nodes_reached = 1

        # frontier가 비어 있지 않은 동안 탐색을 계속합니다 [4].
        while frontier:
          
            print("")
            print("################ POP #################")
            print(f"pop전 frontier상태 : {frontier}")
            # 우선순위 큐에서 가장 낮은 f(n) 값을 가진 노드를 꺼냅니다 [4, 5].
            f_cost, path_cost, current_node = heapq.heappop(frontier)

            print(f'frontier에서 pop한 노드: f={f_cost}, g={path_cost}, {current_node}')
            print("################ POP #################")
            print("")

            nodes_explored += 1

            # 현재 노드의 상태가 목표 상태인지 확인합니다 [4, 6].
            if problem.is_goal_state(current_node.get_state()):
                print(f'목표 상태를 찾았습니다 : {current_node.get_state()}')
                return SearchResults(current_node.path_to_root(), path_cost, nodes_reached, nodes_explored)

            # 현재 상태를 기반으로 가능한 자식 상태들을 생성합니다 [6, 7].
            next_states = problem.generate_children(current_node.get_state())
            print(f'while frontier에서 next state count : {len(next_states)}')

            if next_states:
                for next_state in next_states:
                    print('')
                    # 자식 상태의 표현을 얻습니다 [8, 9].
                    s = next_state.get_representation()

                    # 부모 노드에서 자식 노드로 이동하는 action과 그 비용을 얻습니다 [10, 11].
                    action = next_state.get_location()
                    print(f'action은: {action}')

                    cost = problem.get_action_cost(current_node.get_state(), action)
                    new_path_cost = path_cost + cost

                    # 자식 노드의 휴리스틱 값을 추정합니다 [12, 13].
                    heuristic_cost = problem.estimate_cost_to_solution(next_state)
                    new_f_cost = new_path_cost + heuristic_cost

                    # 자식 노드를 생성합니다 [8, 14].
                    child_node = SearchTreeNode(current_node, action, next_state, new_path_cost)

                    # 만약 자식 상태가 아직 도달하지 않았거나, 현재 경로의 f(n) 값이 이전에 도달했던 f(n) 값보다 작다면 [15]:
                    if s not in reached or new_f_cost < reached[s]:
                        nodes_reached += 1
                        reached[s] = new_f_cost
                        # 우선순위 큐에 (새로운 f(n), 새로운 g(n), 자식 노드) 형태로 추가합니다 [5, 8].
                        heapq.heappush(frontier, (new_f_cost, new_path_cost, child_node))
                        print(f'{s} frontier.heappush 후 frontier 상태 {frontier}')
                        print(f'{s} frontier.heappush 후 reached 상태 {reached}')
                    else:
                        print(f'{s}는 reached에 있으며, 현재 f(n)값{new_f_cost}보다 더 낮은 f(n)값{reached[s]}으로 이미 도달했습니다. frontier 상태: {frontier}')
                        print(f'{s}는 reached에 있으며, 현재 f(n)값{new_f_cost}보다 더 낮은 f(n)값{reached[s]}으로 이미 도달했습니다. reached 상태: {reached}')

        # 목표 상태에 도달하지 못하면 실패를 반환합니다 [8, 16].
        print('탐색 실패')
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

    """
        Auxiliary Function for running the search algorithm specified, 
        and printing the results and statistics.
    """
    @staticmethod
    def search(problem: Problem, algorithm: int):
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

        SearchAlgorithms.print_solution_details(problem.get_current_case(), solution, total_time)

class TreeNode:
    def __init__(self, state_repr, children=None, explored=False):
        self.state_repr = state_repr
        self.children = children if children else []
        self.explored = explored
    
    def add_child(self, child_node):
        self.children.append(child_node)
    
    def mark_explored(self):
        self.explored = True
    
    def find_node(self, state_repr):
        """주어진 상태 표현을 가진 노드를 찾습니다."""
        if self.state_repr == state_repr:
            return self
        
        for child in self.children:
            found = child.find_node(state_repr)
            if found:
                return found
        
        return None
    
    def has_state(self, state_repr):
        """트리에 주어진 상태 표현을 가진 노드가 있는지 확인합니다."""
        return self.find_node(state_repr) is not None


def update_tree(tree_root, current_state_repr, frontier, reached):
    """현재 BFS 상태에 기반하여 트리를 업데이트합니다."""
    # 현재 노드를 탐색됨으로 표시
    node_to_mark = tree_root.find_node(current_state_repr)
    if node_to_mark:
        node_to_mark.mark_explored()
    
    # 자식 노드 추가
    if node_to_mark:
        for child_state in reached:
            if not tree_root.has_state(child_state) and child_state != current_state_repr:
                # frontier에 있는 노드인지 확인
                is_in_frontier = False
                for node in frontier:
                    if node.get_state().get_representation() == child_state:
                        is_in_frontier = True
                        break
                
                if is_in_frontier:
                    node_to_mark.add_child(TreeNode(child_state))
    
    return tree_root


def print_tree(node, indent="", is_last=True, is_root=True):
    """트리 구조를 ASCII 아트로 출력합니다."""
    if is_root:
        print(f"{indent}{node.state_repr} (root){' (explored)' if node.explored else ''}")
    else:
        branch = "└── " if is_last else "├── "
        print(f"{indent}{branch}{node.state_repr}{' (explored)' if node.explored else ''}")
    
    indent += "    " if is_last or is_root else "│   "
    
    for i, child in enumerate(node.children):
        is_last_child = i == len(node.children) - 1
        print_tree(child, indent, is_last_child, False)

