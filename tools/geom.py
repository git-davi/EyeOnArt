import numpy as np


def intersection() :
    pass


def feature_scaling(features) :
    _min = np.amin(features)
    _max = np.amax(features)

    features = ( features - _min ) / ( _max - _min )

    return features, _min, _max

def feature_descaling(features, _min, _max) :
    
    return (features * (_max - _min)) + _min