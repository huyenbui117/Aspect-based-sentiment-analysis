from module.model.models import Model
class chi2Model(Model):
    def __init__(self, model, lableVocab):
        self.vocab=[]
        self.NUM_OF_ASPECTS = len(lableVocab)
        for lable in lableVocab:
            _vocab = []
