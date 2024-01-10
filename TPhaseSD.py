


class TwoPhaseSecDrainage:
    # secondary drainage simulation
    def __new__(cls, obj):
        obj.__class__ = TwoPhaseSecDrainage
        return obj
    
    def __init__(self):
        pass