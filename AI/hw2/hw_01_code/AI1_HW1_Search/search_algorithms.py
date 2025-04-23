
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
        """향상된 디버깅 기능과 트리 시각화를 갖춘 BFS 구현"""
        print("\n===== BFS 알고리즘 시작 =====")
        
        # 시작 노드를 생성합니다.
        initial_state = problem.get_initial_state()
        if initial_state is None:
            return SearchResults(None, None, 0, 0)
        start_node = SearchTreeNode(None, None, initial_state, 0)
        
        # 초기 상태 디버깅 출력
        initial_state_repr = initial_state.get_representation()
        print(f"초기 상태: {initial_state_repr}")
        print(f"시작 위치: {initial_state.get_location()}")
        print(f"초기 방문 상태: {initial_state.get_visited_targets()}")

        # 목표 상태에 도달했는지 확인합니다.
        if problem.is_goal_state(start_node.get_state()):
            return SearchResults(start_node.path_to_root(), start_node.get_path_cost(), 1, 1)

        # FIFO 큐를 사용하여 frontier를 초기화하고 시작 노드를 추가합니다.
        frontier = [start_node]
        # 이미 도달한 상태를 추적하기 위한 집합을 초기화하고 시작 상태를 추가합니다.
        reached = {initial_state_repr}
        
        print(f"\n=== 단계 0: 초기화 ===")
        print(f"frontier: [{initial_state_repr}]")
        print(f"reached: {reached}")
        
        # 트리 초기화 및 시각화
        tree_root = TreeNode(initial_state_repr)
        print("\n트리 상태:")
        print_tree(tree_root)

        # 탐색된 노드와 도달한 노드의 수를 초기화합니다.
        nodes_explored = 0
        nodes_reached = 1
        step_counter = 1

        # frontier가 비어 있지 않은 동안 탐색을 계속합니다.
        while frontier:
            # frontier에서 가장 오래된 노드를 꺼냅니다 (FIFO).
            current_node = frontier.pop(0)
            current_state = current_node.get_state()
            current_state_repr = current_state.get_representation()
            print(f'frontier에서 pop한것 {current_state_repr} reached :{reached}')

            nodes_explored += 1
            
            print(f"\n=== 단계 {step_counter}: {current_state_repr} 노드 확장 ===")
            print(f"현재 위치: {current_state.get_location()}")
            print(f"현재 방문 상태: {current_state.get_visited_targets()}")
            
            # 현재 상태를 기반으로 가능한 자식 상태들을 생성합니다.
            next_states = problem.generate_children(current_node.get_state())
            
            child_added = False
            children_repr = []
            
            if next_states:
                print(f"자식 노드 생성 중... (총 {len(next_states)}개)")
                for next_state in next_states:
                    s = next_state.get_representation()
                    children_repr.append(s)
                    
                    # 자식 상태가 아직 도달하지 않은 상태라면:
                    if s not in reached:
                        nodes_reached += 1
                        reached.add(s)
                        # 새로운 노드를 생성하고 frontier에 추가합니다.
                        action = next_state.get_location()
                        cost = problem.get_action_cost(current_node.get_state(), action)
                        child_node = SearchTreeNode(current_node, action, next_state,
                                                current_node.get_path_cost() + cost)
                                                
                        print(f"  → 새 노드 추가: {s} (이동 위치: {action}, 비용: {cost:.2f})")
                        child_added = True
                        
                        # 자식 노드가 목표 상태인지 확인합니다.
                        if problem.is_goal_state(child_node.get_state()):
                            print(f"\n=== 목표 상태 발견! ===")
                            print(f"목표 상태: {s}")
                            print(f"경로: {child_node.path_to_root()}")
                            print(f"총 비용: {child_node.get_path_cost()}")
                            
                            # 현재 노드에 목표 상태 노드 추가
                            node_to_mark = tree_root.find_node(current_state_repr)
                            if node_to_mark:
                                node_to_mark.add_child(TreeNode(s, [], True))
                            
                            # 최종 트리 상태 출력
                            print("\n최종 트리 상태:")
                            print_tree(tree_root)
                            
                            return SearchResults(child_node.path_to_root(), child_node.get_path_cost(),
                                            nodes_reached, nodes_explored)
                        
                        frontier.append(child_node)
                    else:
                        print(f"  → 이미 방문한 노드: {s}")
            
            # 현재 트리 상태 업데이트 및 시각화
            tree_root = update_tree(tree_root, current_state_repr, frontier, reached)
            
            print("\n현재 트리 상태:")
            print_tree(tree_root)
            
            print(f"\n추가 정보:")
            print(f"frontier: {[node.get_state().get_representation() for node in frontier]}")
            print(f"reached (총 {len(reached)}개): {reached}")
            
            step_counter += 1

        # 목표 상태에 도달하지 못하면 실패를 반환합니다.
        print("\n=== 목표 상태를 찾지 못했습니다 ===")
        return SearchResults(None, None, nodes_reached, nodes_explored)

    # def breadth_first_search(problem: Problem) -> SearchResults:
    #     # 디버깅 로그 추가
    #     print("\n===== BFS 알고리즘 시작 =====")
        
    #     # 시작 노드를 생성합니다.
    #     initial_state = problem.get_initial_state()
    #     if initial_state is None:
    #         return SearchResults(None, None, 0, 0)
    #     start_node = SearchTreeNode(None, None, initial_state, 0)
        
    #     # 초기 상태 디버깅 출력
    #     print(f"초기 상태: {initial_state.get_representation()}")
    #     print(f"시작 위치: {initial_state.get_location()}")
    #     # print(f"목표 위치: {problem.get_targets()}")
    #     print(f"초기 방문 상태: {initial_state.get_visited_targets()}")

    #     # 목표 상태에 도달했는지 확인합니다.
    #     if problem.is_goal_state(start_node.get_state()):
    #         return SearchResults(start_node.path_to_root(), start_node.get_path_cost(), 1, 1)

    #     # FIFO 큐를 사용하여 frontier를 초기화하고 시작 노드를 추가합니다.
    #     frontier = [start_node]
    #     # 이미 도달한 상태를 추적하기 위한 집합을 초기화하고 시작 상태를 추가합니다.
    #     reached = {initial_state.get_representation()}
    #     print(f"\n=== 단계 0: 초기화 ===")
    #     print(f"frontier: [{initial_state.get_representation()}]")
    #     print(f"reached: {reached}")
    #     print("트리 상태:")
    #     print(f"{initial_state.get_representation()} (시작 노드)")

    #     # 탐색된 노드와 도달한 노드의 수를 초기화합니다.
    #     nodes_explored = 0
    #     nodes_reached = 1
    #     step_counter = 1

    #     # frontier가 비어 있지 않은 동안 탐색을 계속합니다.
    #     while frontier:
    #         # frontier에서 가장 오래된 노드를 꺼냅니다 (FIFO).
    #         current_node = frontier.pop(0)
    #         current_state = current_node.get_state()
    #         nodes_explored += 1
            
    #         print(f"\n=== 단계 {step_counter}: {current_state.get_representation()} 노드 확장 ===")
    #         print(f"현재 위치: {current_state.get_location()}")
    #         print(f"현재 방문 상태: {current_state.get_visited_targets()}")
            
    #         # 현재 상태를 기반으로 가능한 자식 상태들을 생성합니다.
    #         next_states = problem.generate_children(current_node.get_state())
            
    #         child_added = False
    #         children_repr = []
            
    #         if next_states:
    #             print(f"자식 노드 생성 중... (총 {len(next_states)}개)")
    #             for next_state in next_states:
    #                 s = next_state.get_representation()
    #                 children_repr.append(s)
                    
    #                 # 자식 상태가 아직 도달하지 않은 상태라면:
    #                 if s not in reached:
    #                     nodes_reached += 1
    #                     reached.add(s)
    #                     # 새로운 노드를 생성하고 frontier에 추가합니다.
    #                     action = next_state.get_location()
    #                     cost = problem.get_action_cost(current_node.get_state(), action)
    #                     child_node = SearchTreeNode(current_node, action, next_state,
    #                                             current_node.get_path_cost() + cost)
                                                
    #                     print(f"  → 새 노드 추가: {s} (이동 위치: {action}, 비용: {cost:.2f})")
    #                     child_added = True
                        
    #                     # 자식 노드가 목표 상태인지 확인합니다.
    #                     if problem.is_goal_state(child_node.get_state()):
    #                         print(f"\n=== 목표 상태 발견! ===")
    #                         print(f"목표 상태: {s}")
    #                         print(f"경로: {child_node.path_to_root()}")
    #                         print(f"총 비용: {child_node.get_path_cost()}")
    #                         return SearchResults(child_node.path_to_root(), child_node.get_path_cost(),
    #                                         nodes_reached, nodes_explored)
    #                     frontier.append(child_node)
    #                 else:
    #                     print(f"  → 이미 방문한 노드: {s}")
            
    #         # 현재 트리 상태 출력
    #         print("\n현재 트리 상태:")
    #         print(f"탐색 완료: {current_state.get_representation()}")
    #         if child_added:
    #             print(f"추가된 자식 노드: {', '.join([c for c in children_repr if c in reached and c not in [node.get_state().get_representation() for node in frontier]])}")
            
    #         print(f"frontier: {[node.get_state().get_representation() for node in frontier]}")
    #         print(f"reached (총 {len(reached)}개): {reached}")
            
    #         step_counter += 1

    #     # 목표 상태에 도달하지 못하면 실패를 반환합니다.
    #     print("\n=== 목표 상태를 찾지 못했습니다 ===")
    #     return SearchResults(None, None, nodes_reached, nodes_explored)

    # def breadth_first_search(problem: Problem) -> SearchResults:
    #     # 시작 노드를 생성합니다.
    #     initial_state = problem.get_initial_state()
    #     if initial_state is None:
    #         return SearchResults(None, None, 0, 0)
    #     start_node = SearchTreeNode(None, None, initial_state, 0)

    #     # 목표 상태에 도달했는지 확인합니다.
    #     if problem.is_goal_state(start_node.get_state()):
    #         return SearchResults(start_node.path_to_root(), start_node.get_path_cost(), 1, 1)

    #     # FIFO 큐를 사용하여 frontier를 초기화하고 시작 노드를 추가합니다 [4, 5].
    #     frontier = [start_node]
    #     # 이미 도달한 상태를 추적하기 위한 집합을 초기화하고 시작 상태를 추가합니다 [5].
    #     reached = {initial_state.get_representation()}
    #     print(f'frontier: {frontier} reached: {reached}')

    #     # 탐색된 노드와 도달한 노드의 수를 초기화합니다.
    #     nodes_explored = 0
    #     nodes_reached = 1

    #     # frontier가 비어 있지 않은 동안 탐색을 계속합니다 [5].
    #     while frontier:
    #         # frontier에서 가장 오래된 노드를 꺼냅니다 (FIFO) [4, 5].
    #         current_node = frontier.pop(0)
    #         nodes_explored += 1

    #         # 현재 상태를 기반으로 가능한 자식 상태들을 생성합니다.
    #         # 이제 problem.generate_children 함수는 다음 가능한 모든 상태를 반환합니다.
    #         next_states = problem.generate_children(current_node.get_state())

    #         if next_states:
    #             for next_state in next_states:
    #                 s = next_state.get_representation()
    #                 print(f'reached s: {s}')

    #                 # 자식 상태가 아직 도달하지 않은 상태라면 [5]:
    #                 if s not in reached:
    #                     nodes_reached += 1
    #                     reached.add(s)
    #                     # 새로운 노드를 생성하고 frontier에 추가합니다 [5].
    #                     # action은 현재 노드의 상태에서 다음 상태로 이동하기 위해 수행된 액션입니다.
    #                     # 이는 generate_children 함수를 통해 생성된 next_state가 어떤 이웃 위치로 이동한 결과인지 알면 됩니다.
    #                     # 따라서, current_node의 위치와 next_state의 위치를 비교하여 action을 파악할 수 있습니다.
    #                     action = next_state.get_location() # 이동한 위치 자체가 action이 됩니다 [1, 2].
    #                     print(f'action: {action}')
    #                     cost = problem.get_action_cost(current_node.get_state(), action)
    #                     child_node = SearchTreeNode(current_node, action, next_state,
    #                                                 current_node.get_path_cost() + cost)
    #                     # 자식 노드가 목표 상태인지 확인합니다 [5].
    #                     if problem.is_goal_state(child_node.get_state()):
    #                         return SearchResults(child_node.path_to_root(), child_node.get_path_cost(),
    #                                                nodes_reached, nodes_explored)
    #                     frontier.append(child_node)

    #     # 목표 상태에 도달하지 못하면 실패를 반환합니다 [5].
    #     return SearchResults(None, None, nodes_reached, nodes_explored)

    """
        Implementation of the Uniform Cost Search (UCS) algorithm. The only input
        required is the problem, which includes a reference to current CityMap and 
        current Search Request. The function needs to compute and store search 
        results and other statistics inside of a SearchResults object. 
    """
    @staticmethod
    def uniform_cost_search(problem: Problem) -> SearchResults:
        # TODO: Your CODE HERE
        return SearchResults(None, None, 0, 0)

    """
        Implementation of the A* Search algorithm. The only input
        required is the problem, which includes a reference to current CityMap and 
        current Search Request. The function needs to compute and store search 
        results and other statistics inside of a SearchResults object. 
    """
    @staticmethod
    def A_start_search(problem: Problem) -> SearchResults:
        # TODO: Your CODE HERE
        return SearchResults(None, None, 0, 0)

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

