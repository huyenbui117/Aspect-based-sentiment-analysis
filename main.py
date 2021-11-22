import os
import pandas as pd
from module.model.lr import PolarityLRModel
from module.preprocess import preprocess, load_data

NUM_OF_ASPECTS = 6

input_data = [
    'data/text.csv',
]
output_data = [
    'data/predict_text.csv'
]
aspectName = ['giá', 'dịch_vụ', 'an_toàn', 'chất_lượng', 'ship', 'chính_hãng']
if __name__ == '__main__':

    model = PolarityLRModel()
    for count, f in enumerate(input_data):
        name = f.split('/')
        input_abs_file_path = f
        output_abs_file_path = output_data[count]

        print(name)
        print('Running {}...'.format(f))

        df = pd.read_csv(input_abs_file_path)
        # print(df)
        for aspectId in range(NUM_OF_ASPECTS):
            print('Runnning {}...'.format(aspectName[aspectId]))
            inputs, outputs = load_data(input_abs_file_path, aspectId)
            inputs = preprocess(inputs)
            file_path = 'save_model/Model_save/lr_model/{}.sav'.format(aspectName[aspectId])
            model.load(file_path, aspectId)
            predicts = model.predict(inputs, aspectId)
            # for count, i in enumerate(inputs):
            #     print(i.text)
            #     print(predicts[count].getall())

            for _, r in df.iterrows():
                for count, i in enumerate(inputs):
                    if r['text'] == i.text:
                        # print(r['text'])
                        # print(predicts[count].scores)
                        df.at[_,'aspect{}'.format(aspectId)] = predicts[count].scores
        print(df)
        df.to_csv(output_abs_file_path)