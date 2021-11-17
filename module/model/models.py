class Input:
    def __init__(self, text):
        self.text = text


class Output:
    def __init__(self, labels, aspects, scores):
        self.labels = labels
        self.aspects = aspects
        self.scores = scores

    def getall(self):
        return [self.aspects, self.labels, self.scores]


class Model:
    def train(self, inputs, outputs):
        """

        :param inputs:
        :param outputs:
        :return:
        """
        raise NotImplementedError

    def save(self, path):
        """

        :param path:
        :return:
        """
        raise NotImplementedError

    def load(self, path):
        """

        :param path:
        :return:
        """
        raise NotImplementedError

    def predict(self, inputs):
        """

        :param inputs:
        :return:
        :rtype: list of models.evaluate
        """
        raise NotImplementedError

    def evaluate_pos(self, y_test, y_predicts):
        raise NotImplementedError

    def evaluate_neg(self, y_test, y_predicts):
        raise NotImplementedError
