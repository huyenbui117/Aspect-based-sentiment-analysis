import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
import os

def feature_select(corpus, labels, k=1000000):
    """
    select features through chi-square test
    :param corpus:
    :param labels:
    :return:
    """
    bin_countVec = CountVectorizer(analyzer="word")
    labelEncoder = LabelEncoder()
    X = bin_countVec.fit_transform(corpus, np.nan)
    y = labelEncoder.fit_transform(labels).reshape(-1, 1)
    selectKBest = SelectKBest(chi2, k="all")
    selectKBest.fit(X, y)

    feature_ids = selectKBest.get_support(indices=True)
    feature_names = bin_countVec.get_feature_names()
    output = {}
    vocab = []

    for new_fId, old_fId in enumerate(feature_ids):
        feature_name = feature_names[old_fId]
        vocab.append(feature_name)
    output['text'] = vocab
    output['_score'] = list(selectKBest.scores_)
    output['_pvalue'] = list(selectKBest.pvalues_)

    return output


input_data = [
    'data/raw_data/mebe_tiki.csv',
]
LABEL = {
    # 'data/raw_data/tech_tiki.csv': 'giá,dịch_vụ,ship,hiệu_năng,chính_hãng,cấu_hình,phụ_kiện,mẫu_mã'
    # 'data_train/tech_shopee.csv': 'giá,dịch_vụ,ship,hiệu_năng,chính_hãng,cấu_hình,phụ_kiện,mẫu_mã',
    'data/raw_data/mebe_tiki.csv': 'aspect0,aspect1,aspect2,aspect3,aspect4,aspect5'
    # 'data_train/mebe_shopee.csv': 'giá,dịch_vụ,an_toàn,chất_lượng,ship,chính_hãng',

}
if __name__ == '__main__':
    for f in input_data:
        print('Running {}...'.format(f))
        label = LABEL[f].split(',')
        print(label)
        for l in label:
            name = f.split('/')
            name = name[2].split('.')
            name = name[0]

            script_path = os.path.dirname(__file__)
            script_dir = os.path.split(script_path)[0]
            script_dir = os.path.split(script_dir)[0]
            abs_file_path = script_dir + '/' + f

            print(name)
            # f_out = open('chi2\\label_{}'.format(l) + '_{}.csv'.format(name), 'w', encoding='utf-8')
            df = pd.read_csv(abs_file_path, encoding='utf-8')
            # df = df[df[l] != 0]
            data = df['text'].astype(str)

            # print(data)
            data_label = df[l]
            # print(df[l])
            data_train = []
            data_train_dict = []
            for k in data:
                data_train.append(k)
            for k1 in data_label:
                data_train_dict.append(abs(k1))

            file_out = feature_select(data_train, data_train_dict)
            df = pd.DataFrame(file_out)
            df = df.sort_values('_score', ascending=False)
            df.to_csv('output/label_{}'.format(l) + '_{}'.format(name), ',', encoding='utf-8')
