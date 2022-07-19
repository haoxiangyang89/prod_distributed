def comp_obj():
    # compute the value of the objective
    from GlobalVariable import globalVariables as gbv
    import numpy as np

    VV = np.sum(gbv.V, axis=2)
    ob = np.sum(np.multiply(gbv.P, gbv.U)) + np.sum(np.multiply(gbv.H, VV))

    return ob