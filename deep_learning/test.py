import pandas as pd
from deep import predict
#create a test csv file
text = ['shop đóng_gói cẩn_thận , giao hàng . bé uống hợp . ủng_hộ', 'Tôi rất hài_lòng về sản_phẩm vì rẻ', 'Sản phẩm khá đắt, chất_lượng ổn']
dict = {'text':text}
df = pd.DataFrame(dict)
df.to_csv('input_test.csv')



#predict for csv
new_data = pd.read_csv('input_test.csv')
list_text = []
for i in range(len(new_data)):
    list_text.append(str(new_data['text'].loc[i]))
print(list_text)
list_label = []
for txt in list_text:
    list_label.append(predict(txt))
gia = []
dich_vu = []
an_toan = []
chat_luong = []
ship = []
other = []
chinh_hang = []

for label in list_label:
    gia.append(label[0])
    dich_vu.append(label[1])
    an_toan.append(label[2])
    chat_luong.append(label[3])
    ship.append(label[4])
    other.append(label[5])
    chinh_hang.append(label[6])

predict_dict = {'text': text, 'giá': gia, 'dịch_vụ': dich_vu, 'an_toàn': an_toan, 'chất_lượng': chat_luong, 'ship': ship, 'other': other, 'chính_hãng': chinh_hang}
pre_df = pd.DataFrame(predict_dict)
pre_df.to_csv('output_test.csv')
