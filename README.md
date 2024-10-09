# Fire Detection

## How to use

* Create virtual environment:
    * `python -m venv venv`
    * `pip install -r requirements.txt`

* Run the code:
    * Download the Sen2Fire dataset using this [link](https://zenodo.org/records/10881058) and place the downloaded `Sen2Fire/` folder under the `datasets/` directory.
    * Use `project/config.yaml` to config the project.
    * To run the project, use the following command: `python3 test.py`.

## Project structure

```
.
├── baseline                    # code from Sen2Fire
│   ├── dataset
│   ├── docs
│   ├── Exp
│   ├── Map
│   ├── model
│   ├── README.md
│   ├── Test.py
│   ├── Train.py
│   └── utils
├── datasets
│   └── Sen2Fire                # dataset from Sen2Fire
│       ├── scene1
│       ├── scene2
│       ├── scene3
│       └── scene4
├── project
│   ├── config.yaml
│   ├── models
│   ├── outputs
│   ├── Sen2FireLoader.py
│   ├── test.py
│   └── utils.py
├── README.md
├── requirements.txt
└── venv
```

## Results

The folder `project/models/` contains all models and records.

* `normal` contains the default models from PyTorch.
* `80by20` contains results using 80% for training and 20% for validation.
