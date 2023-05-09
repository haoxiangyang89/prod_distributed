# class object for global variables
import numpy as np

class global_var:
    def __init__(self, name, keys):
        self.name = name
        self.keys = keys
        self.size = len(keys)
        self.value = np.zeros(self.size)

    def init_dual(self, item_list):
        self.dual = [np.zeros(self.size) for i in item_list]