
# =========================================
#  Created by: Kenny Davila Castellanos
#      For: CSC 480 - AI 1
#
#  TODO: Modified by: ???
#  TODO: Modified When: ???
# =========================================

from .operand import Operand
from .operator import Operator
class OperatorTree:
    def __init__(self, root):
        # this is a "private" attribute of the class
        self.__root = root

    def evaluate(self):
        # TODO: evaluate the expression .. starting from the root
        if self.__root is None:
            raise ValueError("Cannot evaluate empty tree")
        return self.__root.evaluate()

    def post_order_list(self):
        # TODO: create a post-order traversal .. starting from the root
        # HINT: you will need a list to put the results.
        result_list = []
        self.__root.post_order_list(result_list)
        return result_list

    @staticmethod
    def BuildFromJSON(json_data):
        # TODO: This function should create the tree using the provided JSON data
        #       this might need to check the type of the root node because
        #       root could be either an Operator or an Operand
        #
        # TODO: after creating the tree, it should return an instance of this
        #       class (OperatorTree), with the root value properly set up as the
        #       root node of the tree.

        operator_tree = json_data.get("operator_tree")
        if operator_tree:
            type = operator_tree.get("type")
            if type == "number":
                root = Operand.BuildFromJSON(operator_tree)
            elif type == "operator":
                root = Operator.BuildFromJSON(operator_tree)
            else:
                raise ValueError(f"Invalid root node type: {type}")
            return OperatorTree(root)
        else:
            raise ValueError("Missing 'operator_tree' field in JSON data.")

