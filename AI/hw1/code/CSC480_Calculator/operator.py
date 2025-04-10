
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

        a = self.__children[0].evaluate()
        b = self.__children[1].evaluate()

        return eval(f"{a} {self.get_value()} {b}")

        # if self.get_value() == "+":
        #     return a + b
        # elif self.get_value() == "-":
        #     return a - b
        # elif self.get_value() == "*":
        #     return a * b
        # elif self.get_value() == "/":
        #     return a / b 
        # else:
        #     raise ValueError(f"Invalid operator for {Operator.__name__}: {self.get_value()}")

    def post_order_list(self, out_list):
        # Overrides the post_order_list function from parent class.
        # TODO: Should add itself and its children ... all in post-order
        # hint: recursion is needed

        for child in self.__children:
            child.post_order_list(out_list)
        out_list.append(self.get_value())

    @staticmethod
    def BuildFromJSON(json_data):
        # Overrides the BuildFromJSON function from parent class.
        # TODO: Use  JSON data is used to create and return a valid Operator object
        #       which in turn requires recursively creating its children.
        #
        #  This function assumes that json_data contains the info for an Operator Node
        #     and all of its children, and children of its children, etc.

        type_ = json_data.get("type")
        if not isinstance(type_, str):
            raise TypeError(f"Invalid json_data type for {Operator.__name__}: {type_}")
        if type_ != "operator":
            raise ValueError(f"Invalid json_data type for {Operator.__name__}: {type_}")

        value = json_data.get("value")
        if not isinstance(value, str):
            raise TypeError(f"Invalid json_data type for {Operator.__name__}: {value}")
        if (value != "+" and value != "-" and value != "*" and value != "/"):
            raise ValueError(f"Invalid json_data value for {Operator.__name__}: {value}")

        operands = json_data.get("operands")
        if not isinstance(operands, list):
            raise TypeError(f"Invalid operands list for {Operator.__name__}: {operands}")

        children = []
        for operand in operands:
            type_ = operand.get("type")
            if not isinstance(type_, str):
                raise TypeError(f"Invalid json_data type for {Operator.__name__}: {type_}")
            if type_ == "number":
                children.append(Operand.BuildFromJSON(operand))
            elif type_ == "operator":
                children.append(Operator.BuildFromJSON(operand))
            else:
                raise ValueError(f"Invalid json_data value for {Operator.__name__}: {type_}")

        return Operator(value, children) 