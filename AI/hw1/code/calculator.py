
# =========================================
#  Created by: Kenny Davila Castellanos
#      For: CSC 480 - AI 1
#
#  TODO: Modified by: MIN SONG
#  TODO: Modified When: 041125
# =========================================


import sys
import json

from CSC480_Calculator import OperatorTree, Operator, Operand

def stack_based_evaluation(post_order):
    # TODO: this is a simple evaluation algorithm, using a stack
    #       it sequentially reads the mathematical expression in post-fix notation
    #        - every time it finds an operand (A)
    #             it will simply put (A) on the stack
    #        - every time it finds an operator (OP)
    #             it will take 2 items from stack, (B) and (A)
    #             it must compute result by applying operator
    #                (RES) = (A) (OP) (B)
    #             it will save (RES) to the stack
    #             mind that the ORDER OF THE OPERANDS might affect the result
    #       it does it until the end.

    # Remember:
    #    If the expression is valid, only one item will be there at the end
    #    this is the solution, and must be returned
    #
    # HINT: use isinstance function to check the types of the elements on
    #       the post_order list

    # TODO: your logic here
    stack = []
    for item in post_order:
        if isinstance(item, (int, float)):
            stack.append(item)
        elif (item == "+" or item == "-" or item == "*" or item == "/"):
            if len(stack) < 2:
                raise ValueError("len(stack) < 2")

            b = stack.pop()
            a = stack.pop()
            res = eval(f"{a} {item} {b}")

            stack.append(res)
        else:
            raise ValueError(f"Invalid post_order value: {item}")

    if len(stack) != 1:
        raise ValueError(f"Invalid post_order value: {post_order}")

    return stack

def main():
    # handling the command line arguments , given do not change!
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"\tpython {sys.argv[0]} filename")
        return

    # Step 1
    # TODO: Load the JSON data
    #     The actual filename to load should be on sys.argv ...
    #     * position 0 is always the path to the script
    #     * position 1 ... n are your custom command line arguments

    # TODO: Load the input File using the JSON library
    for file in sys.argv[1:]:
        print(f"Current file: {file}")
        try:
            with open(file, 'r') as f:
                json_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File '{file}' not found.")

        # Step 2
        # TODO: Load the expression from file using the OperatorTree.BuildFromJSON function
        # TODO: You must implement the functions
        #       - OperatorTree.BuildFromJSON
        #       - Operand.BuildFromJSON
        #       - Operator.BuildFromJSON
        operator_tree = OperatorTree.BuildFromJSON(json_data)

        # Step 3
        # TODO: Evaluate the expression (using the evaluate function of the OperatorTree class)
        evaluation = operator_tree.evaluate()
        print(f"Evaluation: {evaluation}")

        # Step 4
        # TODO: Generate a list of the elements on the Operator Tree in post-order and print it!
        post_order_list = operator_tree.post_order_list()
        print(f"Post order list: {post_order_list}")

        # Step 5
        # TODO: Evaluate the expression (again) but using the post fix notation and a stack
        #       This must be done by calling stack_based_evaluation
        evaluation = stack_based_evaluation(post_order_list)
        print(f"Stack based evaluation: {evaluation}")
        print(f"\n")

if __name__ == "__main__":
    main()