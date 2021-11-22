# Aspects based sentiment analysis

- [Dataset](https://github.com/phuonglt26/Vietnamese-E-commerce-Dataset)

## Methods

### Feature selection

- We use Scikit-learn as it provide multiple feature selection methods, including Chi-Squared Test. Scikit-learn gives a
  **SelectKBest** class that can be used with various statistical tests. It will rank the features with the statistical
  test that we've determined and select the top **_k_** performing ones (implying that these terms is viewed as more
  relevant to the task than the others).
- In statistics, the Chi-squared test is used to determine whether two categorical variables independent or related. In
  feature selection, the two variables are the observations of the feature and the occurrence of the class. The outcome
  of the test is a test statistic that has a chi-squared distribution and can be clarified or fail to reject the
  assumption or null hypothesis <img src="https://render.githubusercontent.com/render/math?math=H_{0}">that the observed
  and expected frequencies are equal.

- Given a document <img src="https://render.githubusercontent.com/render/math?math=D">, we estimate
  the <img src="https://render.githubusercontent.com/render/math?math=\chi^{2}\)">
  value and rank them by their score:
  <img src="https://render.githubusercontent.com/render/math?math=\chi ^{2}\sum_{t=1}\sum_{c=1}\frac{(O_{t,c}-E_{t,c})^{2}}{E_{t,c}} = N\sum_{t,c}^{}p_{t}p_{c}\left(\frac{(O_{t,c}/N)-p_{t}p_{c}}{p_{t}p_{c}}\right)^{2}">

- where
    - <img src="https://render.githubusercontent.com/render/math?math=\chi^{2}"> = Pearson's cumulative test statistic, which asymptotically approaches a <img src="https://render.githubusercontent.com/render/math?math=\chi^{2}\)"> distribution.
    - <img src="https://render.githubusercontent.com/render/math?math=O_{t,c}"> =   the number of observations of type <img src="https://render.githubusercontent.com/render/math?math=t"> in class <img src="https://render.githubusercontent.com/render/math?math=c">. 
    - <img src="https://render.githubusercontent.com/render/math?math=N"> = total number of observations.
    - <img src="https://render.githubusercontent.com/render/math?math=E_{t,c} = N p_{t,c}"> = the expected (theoretical) count of type <img src="https://render.githubusercontent.com/render/math?math=t"> in class <img src="https://render.githubusercontent.com/render/math?math=c">, asserted by the null hypothesis that the fraction of type <img src="https://render.githubusercontent.com/render/math?math=t"> in class<img src="https://render.githubusercontent.com/render/math?math=p_{t,c}">

- For each feature, a corresponding high <img src="https://render.githubusercontent.com/render/math?math=\chi^{2}\)">
  score indicates that the null hypothesis of independence (meaning the document class has no impact over the term's
  frequency) should be dismissed and the occurrence of the term and class are dependent, therefore we should select the
  feature for classification. In other words, using this method remove the feature that are most likely autonomous of
  class and consequently unessential for classification.

- We use word-level <img src="https://render.githubusercontent.com/render/math?math=\chi^{2}\)"> as calculated above to
  weight the words in the vocabulary, and use that vocabulary to represent data. Then we proceed to filter each aspect's
  vocabulary manually to reduce dimensions used for classfiers.

### Model

LR, in its basic form, uses a logistic function (e.g., sigmoid, tanh) to model a binary dependent variable.

- The prediction score of LR is calculated by formula:
  <img src="https://render.githubusercontent.com/render/math?math=f(x) = \theta(\textbf{w}^{T}\textbf{x})">

  where <img src="https://render.githubusercontent.com/render/math?math=\theta">: logistics function (e.g., sigmoid,
  tanh, etc.)

- The cost function for LR is defined as:
  <img src="https://render.githubusercontent.com/render/math?math=L = \sum_{D}-ylog(y^{'})-(1-y)log(1-y^{'})">

  where
    - <img src="https://render.githubusercontent.com/render/math?math=D">: dataset, containing of labeled tuples
      <img src="https://render.githubusercontent.com/render/math?math=(x,y))">
    - <img src="https://render.githubusercontent.com/render/math?math=y">: the label in a assigned example, which is
      either be 0 or 1.
    - <img src="https://render.githubusercontent.com/render/math?math=y^{'}">: the predicted value, which is between 0
      and 1, given features in <img src="https://render.githubusercontent.com/render/math?math=x">

## Setup

Download [python 3.8.10](https://www.python.org/downloads/release/python-3810/), choose **Installer** versions, tick add
PATH when download process finished

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

- Use argument `--results True` to see the evaluation results with respect to aspect in the terminal
- Example output:

```
p: [0.9879518072289156, 0.8333333333333334, 0.9375, 0.9130434782608695, 0.9653179190751445, 1.0]
r: [0.9761904761904762, 0.8536585365853658, 0.8333333333333334, 0.9230769230769231, 0.9597701149425287, 0.9583333333333334]
f1: [0.9820359281437125, 0.8433734939759037, 0.8823529411764706, 0.9180327868852459, 0.962536023054755, 0.9787234042553191]
micro: (0.9404096834264432, 0.933456561922366, 0.9369202226345084)
macro: (0.9395244229830438, 0.9173937862436601, 0.9278424295819012)
```

## Evaluation

- Our experiments results stored in [score.csv](score.csv)
- **For experiments only**: One can train and see all model training process by runing
  `py evaluate_allmodels.py`

## Inference

- Input text in [data/text.csv](data/text.csv), check the appearance of aspects in the text by 1 or 0 in the aspect<sub>
  i</sub> collumn
- Note: aspect0, aspect1, aspect2, aspect3, aspect4, aspect5 are 'giá', 'dịch vụ', 'an toàn', 'chất lượng', 'ship', '
  chính hãng' respectively.
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
uvicorn app:app --reload
```

- Go to [localhost:8000/docs](http://localhost:8000/docs), click `POST` &rarr; `Try it out` and try to upload data as
  .`xlsx` file formatted as [text.xlsx](data/text.xlsx)
- Click `Execute` to get results
## Web app

- Run (Program Files (x86) if window 32 bits) to disable Google Chrome CORS temporally

```shell
"C:\Program Files\Google\Chrome\Application\chrome.exe" --disable-web-security --disable-gpu --user-data-dir=~/chromeTemp
```
