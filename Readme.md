## Aspects based sentiment analysis

- [Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
- Download and extract the dataset such that train.zip in root directory

## Setup
Download [python 3.8.10](https://www.python.org/downloads/release/python-3810/), choose **Installer** versions, tick add PATH when download process finished

```shell
git clone https://github.com/huyenbui117/CacChuyenDeKHMT
```
In the project directory
```shell
py -m pip install -r requirements.txt
```
## Training
```shell
py evaluate.py 
```
- Example output: 
```
best checkpoint lightning_logs/ckpt1/dog-cat-resnet18-epoch=10-val_loss=0.09.ckpt
```

## Inference

- Input text in [data/text.csv](data/text.csv), check the appearance of aspects in the text by 1 or 0 in the aspect<sub>i</sub> collumn
- Note: aspect0, aspect1, aspect2, aspect3, aspect4, aspect5 are 'giá', 'dịch vụ', 'an toàn', 'chất lượng', 'ship', 'chính hãng' respectively.
- Run
```shell
py main.py
```
- Example output:
```shell
   id                                               text  aspect0  aspect1  aspect2  aspect3  aspect4  aspect5
0   0  giá rẻ, chất lượng tốt, dịch vụ tốt, ship nhan...        1        1        1        1        1        1
1   1  gói bỉm rất cũ bạc_màu và là loại mẫu_mã từ rấ...        0        0       -1       -1        0        0
2   2  hàng rất tốt, đóng gói cẩn thận, ship hơi lâu ...        0        1        0        1       -1        0
3   2  hàng rất tốt, đóng gói cẩn thận, ship hơi lâu ...        0        1        0        1       -1        0
```
- Output are then stored in [predict_text.csv](data/predict_text.csv)
## API
- Start the server:
```shell
uvicorn main:app --reload
```
- Go to [localhost:8000/docs](http://localhost:8000/docs) and try to upload an image of dog or cat at /predict route
