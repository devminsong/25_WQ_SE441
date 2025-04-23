
# =========================================
#  Created by: Kenny Davila Castellanos
#      For: CSC 480 - AI 1
#
#  TODO: Modified by: MIN SONG
#  TODO: Modified When: 041125
# =========================================

from .operator_tree_element import OperatorTreeElement

class Operand(OperatorTreeElement):
    def __init__(self, value):
        super().__init__(value)

    def evaluate(self):
        # Overrides the evaluate function from parent class.
        # TODO: return it's current value
        return self.get_value()

    def post_order_list(self, out_list):
        # Overrides the post_order_list function from parent class.
        # TODO: Should just add itself to the stack
        out_list.append(self.get_value())

    @staticmethod
    def BuildFromJSON(json_data):
        # Overrides the BuildFromJSON function from parent class.
        # TODO: Use JSON data to create a valid Operand Object
        #       this function assumes that json_data only contains the info for an Operand Node
        
        type = json_data.get("type")        
        if not isinstance(type, str):
            raise TypeError(f"Invalid json_data type for {Operand.__name__}: {type}")        
        if type != "number":
            raise ValueError(f"Invalid json_data type for {Operand.__name__}: {type}")
        
        value = json_data.get("value")
        if not isinstance(value, (int, float)):
            raise TypeError(f"Invalid json_data type for {Operand.__name__}: {value}")
        
        return Operand(value)
        

        


