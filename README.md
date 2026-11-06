# Knowledge Distillation for Lightweight Models in Wildfire Segmentation

Research code associated with the [ICPRAM 2026 paper](https://doi.org/10.5220/0014630000004067).
The current scripts use the Landsat-8 ActiveFire dataset for wildfire segmentation.

## Installation

Use Python 3.10 and run all commands from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Dataset preparation

Download the continent ZIPs from the [ActiveFire dataset](https://github.com/pereira-gha/activefire#datasets-and-weights)
into `datasets/Landsat8-ActiveFire/compressed/`, then extract them:

```bash
python scripts/prepare_data.py
```

The extraction script deletes processed ZIPs. The loader also requires
`datasets/Landsat8-ActiveFire/dataset.pkl`: a pandas table with `full_path` and
`masks_voting` columns containing image and mask paths relative to the dataset
directory. CSV splits are provided in
`data/split/`.

## Training

```bash
# U-Net teacher
mkdir -p weights/teacher/
python -m wildfire_kd.train model.name=unet \
  path.output_model=weights/teacher/ path.weights_test=weights/teacher/

# FDDWNet supervised baseline
python -m wildfire_kd.train

# Feature-based KD; replace YOUR_CHECKPOINT.pt with a saved teacher checkpoint
python -m wildfire_kd.kd-train path.teacher_model=weights/teacher/YOUR_CHECKPOINT.pt
```

Settings are in `conf/train.yaml` and `conf/kd.yaml`; W&B is disabled by default.
Both scripts evaluate saved checkpoints after training. The KD script also runs
CUDA benchmarks afterward, which require an NVIDIA GPU.

## How to cite

If you use this code or build on the method, please cite:

```bibtex
@inproceedings{mamede2026wildfirekd,
  author    = {Rafael M. Mamede and Leonardo M. Ferreira and Mansur Mustafin
               and Eduarda Caldeira and Hélder P. Oliveira and Jaime S. Cardoso
               and Ana F. Sequeira},
  title     = {Knowledge Distillation for Lightweight Models in Wildfire Segmentation},
  booktitle = {Proceedings of the 15th International Conference on Pattern Recognition Applications and Methods (ICPRAM 2026)},
  volume    = {1},
  pages     = {187--196},
  year      = {2026},
  publisher = {SCITEPRESS},
  doi       = {10.5220/0014630000004067},
  isbn      = {978-989-758-797-9}
}
```

Machine-readable metadata is available in [CITATION.cff](CITATION.cff).

If you also use the Landsat-8 ActiveFire images or annotations, please cite
[de Almeida Pereira et al. (2021)](https://doi.org/10.1016/j.isprsjprs.2021.06.002),
the dataset publication.

## Code and model licenses

Original project contributions are MIT-licensed. Third-party components retain
their own terms; some upstream permissions remain unresolved. See
[LICENSE](LICENSE) for attribution and exclusions.

## Dataset and publication licenses

The [ActiveFire dataset](https://github.com/pereira-gha/activefire/blob/main/LICENSE)
is **CC BY 4.0**; preserve attribution, the license link, and modification notices.
[USGS Landsat imagery](https://www.usgs.gov/faqs/are-landsat-data-cloud-still-considered-be-within-public-domain)
is public domain; credit USGS as the source. The
[paper](https://publica-rest.fraunhofer.de/server/api/core/bitstreams/def41ed0-2f78-45fd-a8bd-ca8b5b4a9171/content)
is **CC BY-NC-ND 4.0**. These terms are separate from the software license.
