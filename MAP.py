from ML import ML

class MAP(ML):

    # Constructor
    def __init__(self, pc1, pc2):
        """
        Initialize MAP (Maximum A Posteriori) classifier with prior probabilities.

        :param pc1: float, prior probability of class 1
        :param pc2: float, prior probability of class 2
        """
        super().__init__()
        self.ksi = pc2 / pc1
