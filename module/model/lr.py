import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

from module.model.models import Output
from module.model.models import Model


class PolarityLRModel(Model):
    def __init__(self):
        self.NUM_OF_ASPECTS = 6
        self.vocab = []
        labelVocab = ["aspect0", "aspect1", "aspect2", "aspect3", "aspect4", "aspect5"]
        for label in labelVocab:
            _vocab = []
            with open('C:/Users/65905/Desktop/djangoProject/data/embedding/output/label_{}_mebe_tiki'.format(label), encoding="utf-8") as f:
                for l in f:
                    l = l.split(',')
                    _vocab.append(l)
            self.vocab.append(_vocab)
        self.models = [LogisticRegression() for _ in range(self.NUM_OF_ASPECTS)]

    def _represent(self, inputs, aspectId):
        """

        :param list of models.Input inputs:
        :return:
        """
        features = []
        for ip in inputs:
            _features = [v[2] if v[1] in ip.text else 0 for v in
                         self.vocab[aspectId]]
            features.append(_features)
        return np.array(features).astype(np.float)
    def train(self, inputs, outputs, aspectId):
        """

        :param list of models.Input inputs:
        :param list of models.AspectOutput outputs:
        :return:
        """
        X = self._represent(inputs, aspectId)
        ys = [output.scores for output in outputs]
        self.models[aspectId].fit(X, ys)

    def save(self, path, aspectId):
        # save the model to disk
        pickle.dump(self.models[aspectId], open(path, 'wb'))

    def load(self, path, aspectId):
        # load the model from disk
        model = pickle.load(open(path, 'rb'))
        self.models[aspectId] = model

    def predict(self, inputs, aspectId):
        """
        :param inputs:
        :return:
        :rtype: list of models.AspectOutput
        """
        X = self._represent(inputs, aspectId)
        outputs = []
        predicts = self.models[aspectId].predict(X)
        for output in predicts:
            label = 'aspect{}'.format(aspectId) + (' -' if output == -1 else ' +')
            aspect = 'aspect{}'.format(aspectId)
            outputs.append(Output(label, aspect, output))
        return outputs

    def evaluate(self, y_test, y_predicts):
        tp = 0
        fp = 0
        fn = 0
        for g, p in zip(y_test, y_predicts):
            if g.scores == p.scores == 1:
                tp += 1
            elif g.scores == 1:
                fn += 1
            elif p.scores == 1:
                fp += 1
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
        return tp, fp, fn, p, r, f1
