
# =========================================
#  Created by: Kenny Davila Castellanos
#      For: CSC 480 - AI 1
#
#  TODO: Modified by: ???
#  TODO: Modified When: ???
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
        elif item in ["+", "-", "*", "/"]:
            if len(stack) < 2:
                raise ValueError("Not enough operands for operator: {}".format(item))
            b = stack.pop()
            a = stack.pop()
            if item == "+":
                result = a + b
            elif item == "-":
                result = a - b
            elif item == "*":
                result = a * b
            elif item == "/":
                if b == 0:
                    raise ZeroDivisionError("Division by zero")
                result = a / b
            stack.append(result)
        else:
            raise ValueError("Unknown element in post-order list: {}".format(item))

    if len(stack) != 1:
        raise ValueError("Invalid post-order expression: {}".format(post_order))

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
        result_from_tree = operator_tree.evaluate()
        print(f"The result of the expression (from tree evaluation) is: {result_from_tree}")

        # Step 4
        # TODO: Generate a list of the elements on the Operator Tree in post-order and print it!
        post_order_representation = operator_tree.post_order_list()
        print(f"Post-order representation: {post_order_representation}")

        # Step 5
        # TODO: Evaluate the expression (again) but using the post fix notation and a stack
        #       This must be done by calling stack_based_evaluation
        result_from_postfix = stack_based_evaluation(post_order_representation)
        print(f"The result of the expression (from postfix evaluation) is: {result_from_postfix}")
        print(f"\n")

if __name__ == "__main__":
    main()