
# =========================================
#  Created by: Kenny Davila Castellanos
#      For: CSC 480 - AI 1
#
#  TODO: Modified by: ???
#  TODO: Modified When: ???
# =========================================


from .operator_tree_element import OperatorTreeElement
from .operand import Operand

class Operator(OperatorTreeElement):
    def __init__(self, value, children):
        super().__init__(value)
        # this is a "private" attribute of the class
        self.__children = children

    def evaluate(self):
        # Overrides the evaluate function from parent class.
        # TODO: apply the local operator and return the value
        #       - self._value == "+" ?
        #       - self._value == "*" ?
        #       - self._value == "-" ?
        #       - self._value == "/" ?

        # 먼저 모든 자식 노드를 평가합니다
        operand_values = [child.evaluate() for child in self.__children]
        
        # 연산자에 따라 적절한 연산을 수행합니다
        if self._value == "+":
            return operand_values[0] + operand_values[1]
        elif self._value == "-":
            return operand_values[0] - operand_values[1]
        elif self._value == "*":
            return operand_values[0] * operand_values[1]
        elif self._value == "/":
            # 0으로 나누기 체크
            if operand_values[1] == 0:
                raise ValueError("Division by zero")
            return operand_values[0] / operand_values[1]
        else:
            raise ValueError(f"Unknown operator: {self._value}")

    def post_order_list(self, out_list):
        # Overrides the post_order_list function from parent class.
        # TODO: Should add itself and its children ... all in post-order
        # hint: recursion is needed
        for child in self.__children:
            child.post_order_list(out_list)
        out_list.append(self._value)

    @staticmethod
    def BuildFromJSON(json_data):
        # Overrides the BuildFromJSON function from parent class.
        # TODO: Use  JSON data is used to create and return a valid Operator object
        #       which in turn requires recursively creating its children.
        #
        #  This function assumes that json_data contains the info for an Operator Node
        #     and all of its children, and children of its children, etc.

        if json_data["type"] == "operator":
            value = json_data["value"]
            operands = json_data.get("operands", [])
            children = []
            for operand in operands:
                type = operand.get("type")
                if type == "number":
                    children.append(Operand.BuildFromJSON(operand))
                elif type == "operator":
                    children.append(Operator.BuildFromJSON(operand))
                else:
                    raise ValueError(f"Unknown child type: {type} in operator {value}")
            return Operator(value, children)
        else:
            raise ValueError(f"Invalid JSON data for Operator: {json_data}")

