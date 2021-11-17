import os

from module.model.lr import PolarityLRModel
from module.preprocess import preprocess, load_data

NUM_OF_ASPECTS = 6

input_data = [
    'djangoProject/data/text.csv',
]
LABEL = {
    # 'data/raw_data/tech_tiki.csv': 'giá,dịch_vụ,ship,hiệu_năng,chính_hãng,cấu_hình,phụ_kiện,mẫu_mã'
    # 'data_train/tech_shopee.csv': 'giá,dịch_vụ,ship,hiệu_năng,chính_hãng,cấu_hình,phụ_kiện,mẫu_mã',
    'data/raw_data/mebe_tiki.csv': 'aspect0,aspect1,aspect2,aspect3,aspect4,aspect5'
    # 'data_train/mebe_shopee.csv': 'giá,dịch_vụ,an_toàn,chất_lượng,ship,chính_hãng',
}

aspectName = ['giá', 'dịch_vụ', 'an_toàn', 'chất_lượng', 'ship', 'chính_hãng']
if __name__ == '__main__':

    script_path = os.path.dirname(__file__)
    script_dir = os.path.split(script_path)[0]
    model = PolarityLRModel()

    for f in input_data:
        name = f.split('/')
        name = name[2]
        abs_file_path = script_dir + '/' + f

        print(name)
        print('Running {}...'.format(f))

        for aspectId in range(NUM_OF_ASPECTS):
            print('Runnning {}...'.format(aspectId))
            inputs, outputs = load_data(abs_file_path, aspectId)
            inputs = preprocess(inputs)
            file_path = script_path + '/save_model/Model_save/lr_model/{}.sav'.format(aspectName[aspectId])
            model.load(file_path, aspectId)
            predicts = model.predict(inputs, aspectId)
            for count, i in enumerate(inputs):
                print(i.text)
                print(predicts[count].getall())