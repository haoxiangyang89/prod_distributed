import json

class ADMMparams:
    '''
        A setup for the ADMM hyperparameters.
        ADMM hyperparameters should come from a .json file
    '''
    def __init__(self, file_name):
        # initialize the hyperparameters required for the ADMM process with default values
        with open(file_name, 'r') as param_input:
            param_data = json.load(param_input)
            for (param_name, param_value) in param_data.items():
                setattr(self, param_name, param_value)

