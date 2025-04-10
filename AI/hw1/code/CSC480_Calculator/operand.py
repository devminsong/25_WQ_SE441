
# =========================================
#  Created by: Kenny Davila Castellanos
#      For: CSC 480 - AI 1
#
#  TODO: Modified by: ???
#  TODO: Modified When: ???
# =========================================

from .operator_tree_element import OperatorTreeElement

class Operand(OperatorTreeElement):
    def __init__(self, value):
        super().__init__(value)

    def evaluate(self):
        # Overrides the evaluate function from parent class.
        # TODO: return it's current value
        return self._value

    def post_order_list(self, out_list):
        # Overrides the post_order_list function from parent class.
        # TODO: Should just add itself to the stack
        out_list.append(self._value)

    @staticmethod
    def BuildFromJSON(json_data):
        # Overrides the BuildFromJSON function from parent class.
        # TODO: Use JSON data to create a valid Operand Object
        #       this function assumes that json_data only contains the info for an Operand Node

        # if json_data.get("type") == "number":
        #     value = json_data.get("value")
        #     return Operand(value)
        # else:
        #     raise ValueError(f"Invalid JSON data for Operand: {json_data}")
        
        type = json_data.get("type")
        
        if isinstance(type, str) == False:
            raise TypeError(f"Invalid json_data type for {Operand.__name__}: {type}")
        
        if type != "number":
            raise ValueError(f"Invalid json_data type for {Operand.__name__}: {type}")
        
        value = json_data.get("value")
        if isinstance(value, (int, float)) == False:
            raise ValueError(f"Invalid json_data value for {Operand.__name__}: {value}")
        
        return Operand(value)
        

        


