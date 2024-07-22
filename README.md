# Sentinel-2-fire-detection

## How to use:

* `python -m venv venv`
* `pip install -r requirements.txt`
* `python3 Train.py`
* `python3 Test.py --mode 5`

## Project structure:

```
.
├── dataset
│   ├── Sen2Fire_Dataset.py
│   ├── test.txt
│   ├── train.txt
│   └── val.txt
├── Exp
├── Map
├── model
│   └── Networks.py
├── README.md
├── requirements.txt
├── Sen2Fire
│   ├── scene1
│   ├── scene2
│   ├── scene3
│   └── scene4
├── Test.py
├── Train.py
├── utils
│   └── tools.py
└── venv
```