## Aspects based sentiment analysis

- [Dataset](https://github.com/phuonglt26/Vietnamese-E-commerce-Dataset)

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
p: [0.9879518072289156, 0.8333333333333334, 0.9375, 0.9130434782608695, 0.9653179190751445, 1.0]
r: [0.9761904761904762, 0.8536585365853658, 0.8333333333333334, 0.9230769230769231, 0.9597701149425287, 0.9583333333333334]
f1: [0.9820359281437125, 0.8433734939759037, 0.8823529411764706, 0.9180327868852459, 0.962536023054755, 0.9787234042553191]
micro: (0.9404096834264432, 0.933456561922366, 0.9369202226345084)
macro: (0.9395244229830438, 0.9173937862436601, 0.9278424295819012)
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
