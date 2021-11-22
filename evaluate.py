import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential

from module.evaluate import cal_sentiment_prf
from module.model.lr import PolarityLRModel
from module.preprocess import preprocess, load_data

input_data = [
    'data/raw_data/mebe_tiki.csv',
]
LABEL = {
    # 'data/raw_data/tech_tiki.csv': 'giá,dịch_vụ,ship,hiệu_năng,chính_hãng,cấu_hình,phụ_kiện,mẫu_mã'
    # 'data_train/tech_shopee.csv': 'giá,dịch_vụ,ship,hiệu_năng,chính_hãng,cấu_hình,phụ_kiện,mẫu_mã',
    'data/raw_data/mebe_tiki.csv': 'aspect0,aspect1,aspect2,aspect3,aspect4,aspect5'
    # 'data_train/mebe_shopee.csv': 'giá,dịch_vụ,an_toàn,chất_lượng,ship,chính_hãng',
}

aspectName = ['giá', 'dịch_vụ', 'an_toàn', 'chất_lượng', 'ship', 'chính_hãng']
NUM_OF_ASPECTS = 6
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--results', type=bool)
    for f in input_data:
        name = f.split('/')
        name = name[2].split('.')
        name = name[0]

        abs_file_path = f

        print(name)
        print('Running {}...'.format(f))
        tp = []
        fp = []
        fn = []

        model = PolarityLRModel()

        for aspectId in range(NUM_OF_ASPECTS):
            inputs, outputs = load_data(abs_file_path, aspectId)
            inputs = preprocess(inputs)
            Sequential().compile()

            # five_folds_cross_validation(inputs, outputs, model, aspectId=aspectId)
            X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=14)

            model.train(X_train, y_train, aspectId)

            predicts = model.predict(X_test, aspectId)
            # print(predicts[0].getall())
            _tp, _fp, _fn, p, r, f1 = model.evaluate(y_test, predicts)
            # print(p,r,f1)
            tp.append(_tp)
            fp.append(_fp)
            fn.append(_fn)
            file_path = 'save_model/Model_save/lr_model/{}.sav'.format(aspectName[aspectId])
            model.save(file_path, aspectId)

        cal_sentiment_prf(tp, fp, fn, NUM_OF_ASPECTS, verbal=parser.parse_args())



