import pandas as pd
from module.model.models import Input
from module.model.models import Output
# test each aspect:
aspectName = ['giá', 'dịch_vụ', 'an_toàn', 'chất_lượng', 'ship', 'chính_hãng']
def load_data(path, aspectId):
    """

    :param path:
    :return:
    :rtype: list of models.Input
    """
    inputs = []
    outputs = []
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        if r['aspect{}'.format(aspectId)] != 0:
            t = r['text'].strip()
            inputs.append(Input(t))
            score = r['aspect{}'.format(aspectId)]
            label = aspectName[aspectId] + (' -' if score == -1 else ' +')
            aspect = 'aspect{}'.format(aspectId)
            outputs.append(Output(label, aspect, score))
    return inputs, outputs


def preprocess(inputs):
    return inputs