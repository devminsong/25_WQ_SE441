
# =========================================
#  Created by: Kenny Davila Castellanos
#      For: CSC 480 - AI 1
#
#  TODO: Modified by: MIN SONG
#  TODO: Modified When: 041125
# =========================================

from .operand import Operand
from .operator import Operator
class OperatorTree:
    def __init__(self, root):
        # this is a "private" attribute of the class
        self.__root = root

    def evaluate(self):
        # TODO: evaluate the expression .. starting from the root
        return self.__root.evaluate()

    def post_order_list(self):
        # TODO: create a post-order traversal .. starting from the root
        # HINT: you will need a list to put the results.
        out_list = []
        self.__root.post_order_list(out_list)
        return out_list
        
    @staticmethod
    def BuildFromJSON(json_data):
        operator_tree = json_data.get("operator_tree")
        if not isinstance(operator_tree, dict):
            raise TypeError(f"Invalid json_data type for {OperatorTree.__name__}: {operator_tree}")

        type_ = operator_tree.get("type")
        if not isinstance(type_, str):
            raise TypeError(f"Invalid json_data type for {OperatorTree.__name__}: {type_}")
        
        if type_ == "number":
            root = Operand.BuildFromJSON(operator_tree)
        elif type_ == "operator":
            root = Operator.BuildFromJSON(operator_tree)
        else:
            raise ValueError(f"Invalid json_data type for {OperatorTree.__name__}: {type_}")

        return OperatorTree(root)